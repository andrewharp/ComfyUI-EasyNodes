import { app } from "/scripts/app.js";
import { createSetting } from "./config_service.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

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
    ctx.fillText("ℹ️", node.size[0] - titleHeight - 20,
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
        if (isInsideRectangle(localPos[0], localPos[1], this.size[0] - LiteGraph.NODE_TITLE_HEIGHT,
          -LiteGraph.NODE_TITLE_HEIGHT, LiteGraph.NODE_TITLE_HEIGHT, LiteGraph.NODE_TITLE_HEIGHT)) {
          window.open(this.sourceLoc, "_blank");
        }
      };
    }
  },
});


const origProcessMouseMove = LGraphCanvas.prototype.processMouseMove;
LGraphCanvas.prototype.processMouseMove = function(e) {
  const res = origProcessMouseMove.apply(this, arguments);

  var node = this.graph.getNodeOnPos(e.canvasX,e.canvasY,this.visible_nodes);

  if (!node) {
    return res;
  }
  
  if (!this.canvas) {
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
  if (node.link && !node.flags.collapsed && isInsideRectangle(e.canvasX, e.canvasY, linkX, linkY, linkWidth, linkHeight)) {
      this.canvas.style.cursor = "pointer";
  } else if (desc && isInsideRectangle(e.canvasX, e.canvasY, infoX, infoY, infoWidth, infoHeight)) {
      this.canvas.style.cursor = "help";
      this.tooltip_text = desc;
      this.tooltip_pos = [e.canvasX, e.canvasY];
      this.dirty_canvas = true;
  } 

  return res;
};
