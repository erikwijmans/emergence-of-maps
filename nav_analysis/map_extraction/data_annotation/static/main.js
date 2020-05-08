var post_json = function(url, data, cb) {
    return $.ajax({
      url: url,
      dataType: 'json',
      type: 'POST',
      contentType: 'application/json;charset=UTF-8',
      data: JSON.stringify(data),
      success: cb,
      error: function(err) {
        return console.log(err);
      }
    });
  };



class Annotator {
  set_and_draw_data(data, start_loop = false) {
    this.scroll_position = 0;
    this.data = data;
    this.draw_map();
    console.log(data.path_difference);
    
    if (start_loop) {
      window.requestAnimationFrame(() => this.draw_data());
    }
  }

  draw_path(positions) {
    var prev_xy = positions[0];
    this.ctx.beginPath();
    for (var xy of positions) {
      this.ctx.moveTo(prev_xy[1]* this.cell_size, prev_xy[0]*this.cell_size);

      this.ctx.lineTo(xy[1]*this.cell_size, xy[0]*this.cell_size);

      prev_xy = xy;
    }
    this.ctx.stroke();
  };

  draw_map() {
    var ctx = this.ctx;
    var grid = this.data.grid;
    for (var j = 0; j < grid.length; ++j) {
      for (var i = 0; i < grid[j].length; ++i) {
        if (grid[j][i] == 0) {
          ctx.fillStyle = "#ecf0f1";
        } else {
          ctx.fillStyle = "#95a5a6";
        }

        this.draw_rect(i, j);
      }
    }

    
  };

  draw_data () {
    var ctx = this.ctx;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#9b59b67d";
    this.draw_path(this.data.path2);

    ctx.strokeStyle = "rgba(28, 90, 130, 125)";
    this.draw_path(this.data.path1.slice(this.scroll_position, this.data.path1.length));

    ctx.strokeStyle= "#3498dbff";
    this.draw_path(this.data.path1.slice(0, this.scroll_position + 1));

    ctx.strokeStyle = "#f1c40f";
    this.draw_path(this.data.path1.slice(this.scroll_position-1, this.scroll_position + 1));


    ctx.fillStyle = "#2ecc717d";
    this.draw_rect(this.data.goal[1], this.data.goal[0])

    var styles = ["#e74c3c", "#e67e22"]
    var i = 0;
    for (var idx of this.data.excursions) {
      ctx.fillStyle = styles[i];
      var pt = this.data.path1[idx];
      this.draw_rect(pt[1], pt[0])

      i = (i + 1) % 2;
    }

    window.requestAnimationFrame(() => this.draw_data());
  };

  draw_rect(x, y) {
    this.ctx.fillRect(x*this.cell_size, y*this.cell_size,this.cell_size, this.cell_size);
  }

  on_scroll(event) {
    if (event.originalEvent.wheelDelta > 0) {
      this.scroll_position = Math.max(this.scroll_position - 1, 0);
    } else {
      this.scroll_position = Math.min(this.scroll_position + 1, this.data.path1.length);
    }
  }

  on_jk(event) {
    var do_update = false;
    if (event.originalEvent.code == "KeyJ") {
      this.update_dir = -1;
      do_update = true;
    } else if (event.originalEvent.code == "KeyK") {
      this.update_dir = 1;
      do_update = true;
    }

    if (do_update) {
      this.idx = this.idx + this.update_dir;
      console.log(this.idx);
      post_json("/api/get-task-data", {idx: this.idx}, (data) => this.set_and_draw_data(data));
    }
  }

  on_enter(event) {
    if (event.originalEvent.code == "Enter") {
      this.data.excursions.push(this.scroll_position);
      console.log(this.data.excursions);
    }
  }

  on_save(event) {
    if (event.originalEvent.code == "KeyS") {
      post_json("/api/save-result", {idx: this.idx, excursions: this.data.excursions}, (res) => console.log("Saved!"));

      this.idx = this.idx + this.update_dir;
      console.log(this.idx);
      post_json("/api/get-task-data", {idx: this.idx}, (data) => this.set_and_draw_data(data));
    }
  }

  on_undo(event) {
    if (event.originalEvent.code == "KeyZ") {
      this.data.excursions.pop();
      console.log(this.data.excursions);
    }
  }

  constructor() {
    this.canvas = $("#canvas");
    this.ctx = this.canvas[0].getContext("2d");
    this.excursions = [];
    this.cell_size = 4;
    this.idx = parseFloat(window.idx);
    this.update_dir = 1;

    $(window).on("keypress", (event) => this.on_jk(event));
    $(window).on("keypress", (event) => this.on_enter(event));
    $(window).on("keypress", (event) => this.on_save(event));
    $(window).on("keypress", (event) => this.on_undo(event));

    this.canvas.on("wheel", (event) => this.on_scroll(event));

    this.scroll_position = 0.0;

    post_json("/api/get-task-data", {idx: window.idx}, (data) => this.set_and_draw_data(data, true));
  };

};


var runner = new Annotator();




