# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Gossipers

:description: Gossiper's are designed for multi-peer communication (i.e., send
              and recv from multiple peers at each ieration)
"""

import socket
import threading
import time
from typing import List

import ifcfg
import msgpack_numpy
import torch
import torch.distributed as distrib
import zmq

from .graph_manager import DynamicDirectedExponentialGraph, GraphManager


def _serialize(t):
    return msgpack_numpy.packb(t.numpy(), use_bin_type=True)


def _deserialize(msg):
    return torch.from_numpy(msgpack_numpy.unpackb(msg, raw=False))


class PushSumGossipImpl:
    @classmethod
    def init(
        cls,
        graph,
        gossip_flag,
        train_flag,
        gossip_lock,
        gossip_device_buffer,
        edge_weight,
        gossip_stream,
    ):
        impl = cls(
            graph,
            gossip_flag,
            train_flag,
            gossip_lock,
            gossip_device_buffer,
            edge_weight,
            gossip_stream,
        )
        impl.run()

    def __init__(
        self,
        graph,
        gossip_flag,
        train_flag,
        gossip_lock,
        gossip_device_buffer,
        edge_weight,
        gossip_stream,
    ):
        self.rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.train_flag = train_flag
        self.gossip_flag = gossip_flag
        self.gossip_lock = gossip_lock
        self.gossip_device_buffer = gossip_device_buffer
        self.gossip_stream = gossip_stream
        self.placeholder = gossip_device_buffer.clone()
        self.msg_buffer = gossip_device_buffer.clone()
        self.edge_weight = edge_weight

        self._graph_manager = graph
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(rotate=False)  # sets in- and out-peers attributes

    def refresh_peers_(self, rotate=None):
        """ Update in- and out-peers """
        if rotate is None:
            rotate = True if self._graph_manager.is_dynamic_graph() else False
        # cannot cycle peers in a static graph
        assert not (rotate and not self._graph_manager.is_dynamic_graph())
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)

        self.out_msg_buffer = []

    def run(self):
        self.gossip_flag.set()
        while True:
            self.train_flag.wait()

            with torch.cuda.stream(self.gossip_stream):
                self.mix()

            self.gossip_stream.synchronize()

            self.train_flag.clear()
            self.gossip_flag.set()

    def clear_msgs_(self):
        while len(self.out_msg_buffer) > 0:
            rq = self.out_msg_buffer.pop()
            rq.wait()

    def mix(self):
        with self.gossip_lock:
            self.edge_weight.fill_(1.0 / (len(self.out_edges) + 1.0))
            self.msg_buffer.copy_(self.gossip_device_buffer)
            self.msg_buffer.mul_(self.edge_weight)
            for i, edge in enumerate(self.out_edges):
                assert edge.src == self.rank
                rq = distrib.isend(self.msg_buffer, dst=edge.dest)
                self.out_msg_buffer.append(rq)

            self.gossip_device_buffer.zero_()
            self.gossip_device_buffer[-1] = self.edge_weight
            for i, edge in enumerate(self.in_edges):
                assert edge.dest == self.rank
                distrib.recv(self.placeholder, src=edge.src)
                self.gossip_device_buffer.add_(self.placeholder)

            self.gossip_device_buffer[0:-1].div_(self.gossip_device_buffer[-1])

        self.clear_msgs_()
        self.refresh_peers_(rotate=True)


class Gossiper(object):
    """ Generic gossip averaging object for multi-peer communication """

    def __init__(self, initial_msg, graph=None, prev_weight=1.0):
        """
        Initialize generic averaging class designed for multi-peer comms

        :param graph: (GraphManager) Subclass of GraphManager
        """

        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()
        self.gossip_stream = torch.cuda.Stream()

        assert distrib.is_initialized()
        rank = distrib.get_rank()
        world_size = distrib.get_world_size()

        # graph topology properties
        self.rank = rank
        self.world_size = world_size

        if graph is None:
            graph = DynamicDirectedExponentialGraph(rank, world_size)

        assert isinstance(graph, GraphManager)
        total_numel = sum(ele.numel() for ele in initial_msg) + 1
        self.gossip_device_buffer = torch.empty(
            total_numel, device=initial_msg[0].device, dtype=initial_msg[0].dtype
        )
        self.gossip_device_buffer[-1] = prev_weight
        self.edge_weight = torch.tensor(
            0.0,
            device=self.gossip_device_buffer.device,
            dtype=self.gossip_device_buffer.dtype,
        )

        self._impl_thread = threading.Thread(
            target=PushSumGossipImpl.init,
            args=(
                graph,
                self.gossip_flag,
                self.train_flag,
                self.gossip_lock,
                self.gossip_device_buffer,
                self.edge_weight,
                self.gossip_stream,
            ),
        )
        self._impl_thread.daemon = True
        self._impl_thread.name = "gossiper"
        self._impl_thread.start()

        self.gossip_flag.wait()
        self.gossip_flag.clear()

        self.gossiping = False

    def begin_gossip(self, msg):
        assert not self.gossiping
        self.gossip_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.gossip_stream):
            with self.gossip_lock:
                ptr = 0
                for ele in msg:
                    self.gossip_device_buffer[ptr : ptr + ele.numel()].copy_(
                        ele.view(-1), non_blocking=True
                    )
                    ptr += ele.numel()

        self.gossiping = True
        self.train_flag.set()

    def finish_gossip(self, msg: List[torch.Tensor]):
        if not self.gossiping:
            return

        self.gossip_flag.wait()
        assert self.gossip_flag.is_set()

        with self.gossip_lock:
            ptr = 0
            ele_mul_factor = self.edge_weight / self.gossip_device_buffer[-1]
            for ele in msg:
                ele.mul_(ele_mul_factor)
                ele.add_(
                    self.gossip_device_buffer[ptr : ptr + ele.numel()].view_as(ele)
                )

                ptr += ele.numel()

        self.gossip_flag.clear()
        self.gossiping = False
