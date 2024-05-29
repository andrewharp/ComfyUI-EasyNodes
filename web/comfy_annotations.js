import { app } from "/scripts/app.js";
import { api } from '/scripts/api.js';

app.registerExtension({
  name: "ComfyAnnotations",

  async setup() {
    app.ui.settings.addSetting({
      id: "Comfy.SourcePathPrefix",
      name: "Stack trace remove prefix (common prefix to remove, e.g '/home/user/project/')",
      type: "text",
      defaultValue: "",
    });

    app.ui.settings.addSetting({
      id: "Comfy.EditorPathPrefix",
      name: "Stack trace link prefix (insert this in stack traces to make them clickable, e.g. 'vscode://vscode-remote/wsl+Ubuntu')",
      type: "text",
      defaultValue: "",
      onChange: (value) => {
        console.log("Changed editor path prefix");
      }
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


app.registerExtension({
  name: "ComfyUI-Annotations-ImagePreview",

  nodeCreated(node, app) {
    if (node.comfyClass == "AnythingCache") {
      node._imgs = [new Image()];
      node.imageIndex = 0;

      let already_tried = false;

      Object.defineProperty(node, 'imgs', {
        set(v) {
          console.log("Setting image: ", v);
          if (v && v.length == 0) {
            console.log("No images!");
            return;
          }
          node._imgs = v;
        },
        get() {
          if (!already_tried) {
            already_tried = true;
            try {
              if (node._imgs[0].src == '') {
                const subfolder = "ComfyUI-Annotations";
                const type = "temp";
                const filename = `preview-${node.id}.png`;

                let params = `?filename=${filename}&type=${type}&subfolder=${subfolder}`;

                api.fetchApi('/view/validate' + params, { cache: "no-store" }).then(response => {
                  if (response.status == 200) {
                    console.log("Got image!");
                    node._imgs[0].src = 'view' + params;
                  }
                });
              }
            } catch (e) {
            }
          }

          return node._imgs;
        }
      });
    }
  }
})
