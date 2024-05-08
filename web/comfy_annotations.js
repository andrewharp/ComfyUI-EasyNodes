import { app } from "/scripts/app.js";
import { api } from '/scripts/api.js';

app.registerExtension({
  name: "ComfyAnnotations",

  async setup() {
    app.ui.settings.addSetting({
      id: "Comfy.SourcePathPrefix",
      name: "Stack trace remove prefix (common prefix to remove, e.g '/home/user/project/')",
      type: "string",
      defaultValue: "",
    });
    
    app.ui.settings.addSetting({
      id: "Comfy.EditorPathPrefix",
      name: "Stack trace link prefix (insert this in stack traces to make them clickable, e.g. 'vscode://vscode-remote/wsl+Ubuntu')",
      type: "string",
      defaultValue: "",
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.description) {
      const lines = nodeData.description.split('\n');
      var fgColor = null;
      var bgColor = null;
      var nodeSource = null;
      var outputLines = [];
      
      for (const line of lines) {
        if (line.startsWith('ComfyUINodeColor=')) {
          const color = line.split('=')[1];
          fgColor = color;
        } else if (line.startsWith('ComfyUINodeBgColor=')) {
          const color = line.split('=')[1];
          bgColor = color;
        } else if (line.startsWith('NodeSource=')) {
          const source = line.split('=')[1];
          nodeSource = source;
        } else {
          outputLines.push(line);
        }
      }
      
      nodeData.description = outputLines.join('\n');
      const editorPathPrefix = app.ui.settings.getSettingValue("Comfy.EditorPathPrefix");
      
      function applyColorsAndSource() {
        if (fgColor) {
          this.color = fgColor;
        }
        if (bgColor) {
          this.bgcolor = bgColor;
        }
        if (nodeSource && editorPathPrefix) {
          this.sourceLoc = editorPathPrefix + nodeSource;
        }
        this.description = nodeData.description;
      }
      
      // Apply colors and source location when the node is created
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        applyColorsAndSource.call(this);
      };
      
      // Apply colors and source location when configuring the node
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        applyColorsAndSource.call(this);
      };
    }
  },
});
