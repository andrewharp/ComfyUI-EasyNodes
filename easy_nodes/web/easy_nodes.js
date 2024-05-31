import { app } from '../../scripts/app.js'
import { ComfyWidgets } from "../../scripts/widgets.js";
import { createSetting } from "./config_service.js";

const editorPathPrefixId = "easy_nodes.EditorPathPrefix";

function resizeShowValueWidgets(node, numValues, app) {
  const numShowValueWidgets = (node.showValueWidgets?.length ?? 0);
  numValues = Math.max(numValues, 0);

  if (numValues > numShowValueWidgets) {
    for (let i = numShowValueWidgets; i < numValues; i++) {
      const showValueWidget = ComfyWidgets["STRING"](node, `output${i}`, ["STRING", { multiline: true }], app).widget;
      showValueWidget.inputEl.readOnly = true;
      if (!node.showValueWidgets) {
        node.showValueWidgets = [];
      }
      node.showValueWidgets.push(showValueWidget);
    }
  } else if (numValues < numShowValueWidgets) {
    const removedWidgets = node.showValueWidgets.splice(numValues);
    node.widgets.splice(node.origWidgetCount + numValues);

    // Remove the detached widgets from the DOM
    removedWidgets.forEach(widget => {
      widget.inputEl.parentNode.removeChild(widget.inputEl);
    });
  }
}


function renderSourceLinkAndInfo(node, ctx, titleHeight) {
  if (node.sourceLoc) {
    const link = node.sourceLoc;
    const linkText = "src";
    ctx.fillStyle = "#2277FF";
    ctx.fillText(
      linkText,
      node.size[0] - titleHeight,
      LiteGraph.NODE_TITLE_TEXT_Y - titleHeight
    );
    node.linkWidth = ctx.measureText(linkText).width;
    node.link = link;
  }
  if (node.description?.trim()) {
    ctx.fillText("ℹ️", node.size[0] - titleHeight - node.linkWidth,
      LiteGraph.NODE_TITLE_TEXT_Y - titleHeight);
  }
}

function isInsideRectangle(x, y, left, top, width, height) {
  if (left < x && left + width > x && top < y && top + height > y) {
    return true;
  }
  return false;
}


app.registerExtension({
  name: "EasyNodes",
  async setup() {
    createSetting(
      "easy_nodes.SourcePathPrefix",
      "🪄 Stack trace remove prefix (common prefix to remove, e.g '/home/user/project/')",
      "text",
      ""
    );
    createSetting(
      editorPathPrefixId,
      "🪄 Stack trace link prefix (makes stack traces clickable, e.g. 'vscode://vscode-remote/wsl+Ubuntu')",
      "text",
      ""
    );
    createSetting(
      "easy_nodes.ReloadOnEdit",
      "🪄 Auto-reload EasyNodes source files on edits.",
      "boolean",
      false,
    );
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const easyNodesJsonPrefix = "EasyNodesInfo=";
    if (nodeData?.description.startsWith(easyNodesJsonPrefix)) {
      // EasyNodes metadata will be crammed into the first line of the description in json format.
      const [nodeInfo, ...descriptionLines] = nodeData.description.split('\n');
      const { color, bgColor, sourceLocation } = JSON.parse(nodeInfo.replace(easyNodesJsonPrefix, ""));

      nodeData.description = descriptionLines.join('\n');

      const editorPathPrefix = app.ui.settings.getSettingValue(editorPathPrefixId);

      function applyColorsAndSource() {
        if (color) {
          this.color = color;
        }
        if (bgColor) {
          this.bgcolor = bgColor;
        }
        if (sourceLocation && editorPathPrefix) {
          this.sourceLoc = editorPathPrefix + sourceLocation;
        }
        this.description = nodeData.description;
      }

      // Apply colors and source location when the node is created
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        applyColorsAndSource.call(this);
        this.origWidgetCount = this.widgets?.length ?? 0;
        this.linkWidth = 20;
      };

      // Apply colors and source location when configuring the node
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        applyColorsAndSource.call(this);

        this.origWidgetCount = this.widgets?.length ?? 0;
        const widgetValsLength = this.widgets_values?.length ?? 0;

        const numShowVals = widgetValsLength - this.origWidgetCount;
        resizeShowValueWidgets(this, numShowVals, app);

        for (let i = 0; i < numShowVals; i++) {
          this.showValueWidgets[i].value = this.widgets_values[this.origWidgetCount + i];
        }
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, [message]);

        const numShowVals = message.text.length;

        // console.log(this.id, "onExecuted", numShowVals, message.text.length, this.origWidgetCount);

        resizeShowValueWidgets(this, numShowVals, app);

        for (let i = 0; i < numShowVals; i++) {
          this.showValueWidgets[i].value = message.text[i];
        }

        this.setSize(this.computeSize());
        this.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);
      }

      const onDrawForeground = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx, canvas, graphMouse) {
        onDrawForeground?.apply(this, arguments);
        renderSourceLinkAndInfo(this, ctx, LiteGraph.NODE_TITLE_HEIGHT); 
      };


      const onDrawBackground = nodeType.prototype.onDrawBackground;
      nodeType.prototype.onDrawBackground = function (ctx, canvas) {
        onDrawBackground?.apply(this, arguments);

      }

      const onMouseDown = nodeType.prototype.onMouseDown;
      nodeType.prototype.onMouseDown = function (e, localPos, graphMouse) {
        onMouseDown?.apply(this, arguments);
        // console.log("onMouseDown", this.link, localPos);
        if (this.link && !this.flags.collapsed && isInsideRectangle(localPos[0], localPos[1], this.size[0] - this.linkWidth,
          -LiteGraph.NODE_TITLE_HEIGHT, this.linkWidth, LiteGraph.NODE_TITLE_HEIGHT)) {
          window.open(this.link, "_blank");
        }
      };
    }
  },
});


const origProcessMouseMove = LGraphCanvas.prototype.processMouseMove;
LGraphCanvas.prototype.processMouseMove = function(e) {
  const res = origProcessMouseMove.apply(this, arguments);

  var node = this.graph.getNodeOnPos(e.canvasX,e.canvasY,this.visible_nodes);

  if (!node || !this.canvas || node.flags.collapsed) {
    return res;
  }

  var infoWidth = 20;
  var infoHeight = LiteGraph.NODE_TITLE_HEIGHT;

  var linkWidth = node.linkWidth * 2;

  var linkX = node.pos[0] + node.size[0] - linkWidth;
  var linkY = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT;

  var infoX = linkX - 20;
  var infoY = linkY;
  var infoWidth = 20;
  var infoHeight = LiteGraph.NODE_TITLE_HEIGHT;
  var linkHeight = LiteGraph.NODE_TITLE_HEIGHT;

  const desc = node.description?.trim();
  if (node.link && isInsideRectangle(e.canvasX, e.canvasY, linkX, linkY, linkWidth, linkHeight)) {
      this.canvas.style.cursor = "pointer";
      this.tooltip_text = node.link;
      this.tooltip_pos = [e.canvasX, e.canvasY];
      this.dirty_canvas = true;
  } else if (desc && isInsideRectangle(e.canvasX, e.canvasY, infoX, infoY, infoWidth, infoHeight)) {
      this.canvas.style.cursor = "help";
      this.tooltip_text = desc;
      this.tooltip_pos = [e.canvasX, e.canvasY];
      this.dirty_canvas = true;
  } else {
      this.tooltip_text = null;
  }

  return res;
};


LGraphCanvas.prototype.drawNodeTooltip = function(ctx, text, pos) {
    if (text === null) return;
            
    ctx.save();
    ctx.font = "14px Consolas, 'Courier New', monospace";
    
    var lines = text.split('\n');
    var lineHeight = 18;
    var totalHeight = lines.length * lineHeight;
    
    var w = 0;
    for (var i = 0; i < lines.length; i++) {
        var info = ctx.measureText(lines[i].trim());
        w = Math.max(w, info.width);
    }
    w += 20;
    
    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 2;
    ctx.shadowBlur = 5;
    
    ctx.fillStyle = "#2E2E2E";
    ctx.beginPath();
    ctx.roundRect(pos[0] - w / 2, pos[1] - 15 - totalHeight, w, totalHeight, 5, 5);
    ctx.moveTo(pos[0] - 10, pos[1] - 15);
    ctx.lineTo(pos[0] + 10, pos[1] - 15);
    ctx.lineTo(pos[0], pos[1] - 5);
    ctx.fill();
    
    ctx.shadowColor = "transparent";
    ctx.textAlign = "left";
    
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        
        // Render the colored line
        var el = document.createElement('div');
        
        el.innerHTML = line;
        
        var parts = el.childNodes;
        var x = pos[0] - w / 2 + 10;
        
        for (var j = 0; j < parts.length; j++) {
            var part = parts[j];
            ctx.fillStyle = "#E4E4E4";
            ctx.fillText(part.textContent, x, pos[1] - 15 - totalHeight + (i + 0.8) * lineHeight);
            x += ctx.measureText(part.textContent).width;
        }
    }
    
    ctx.restore();
};

const origdrawFrontCanvas = LGraphCanvas.prototype.drawFrontCanvas;
LGraphCanvas.prototype.drawFrontCanvas = function() {
  origdrawFrontCanvas.apply(this, arguments);
  if (this.tooltip_text) {
    console.log("draw tooltip", this.tooltip_text, this.tooltip_pos);
    this.ctx.save();
    this.ds.toCanvasContext(this.ctx);
    this.drawNodeTooltip(this.ctx, this.tooltip_text, this.tooltip_pos);
    this.ctx.restore();
  }  
};
