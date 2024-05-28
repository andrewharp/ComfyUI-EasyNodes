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


// function getOutputNodesFromSelected(canvas) {
//     return ((canvas.selected_nodes &&
//         Object.values(canvas.selected_nodes).filter((n) => {
//             var _a;
//             return (n.mode != LiteGraph.NEVER &&
//                 ((_a = n.constructor.nodeData) === null || _a === void 0 ? void 0 : _a.output_node));
//         })) ||
//         []);
// }

// function showQueueNodesMenuIfOutputNodesAreSelected(existingOptions, node) {
//     if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") != false) {
//         const canvas = app.canvas;
//         const outputNodes = getOutputNodesFromSelected(canvas);
//         const menuItem = {
//             content: `Queue Selected Output Nodes (rgthree) &nbsp;`,
//             className: "rgthree-contextmenu-item",
//             callback: () => {
//                 rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
//             },
//             disabled: !outputNodes.length,
//         };
//         let idx = existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Outputs") + 1;
//         idx = idx || existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Align") + 1;
//         idx = idx || 3;
//         existingOptions.splice(idx, 0, menuItem);
//     }
//     return existingOptions;
// }

// app.registerExtension({
//     name: "open_file_in_editor",
//     async beforeRegisterNodeDef(nodeType, nodeData) {
//         const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
//         nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
//             getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
//             showQueueNodesMenuIfOutputNodesAreSelected(options, this);
//         };
//     },
//     async setup() {
//         console.log("rgthree-menu-setup");
//         const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
//         LGraphCanvas.prototype.getCanvasMenuOptions = function (...args) {
//             const options = getCanvasMenuOptions.apply(this, [...args]);
//             showQueueNodesMenuIfOutputNodesAreSelected(options);
//             return options;
//         };
//     },
// });



// app.registerExtension({
//     name: "ComfyUI-Annotations",
//     async beforeRegisterNodeDef(nodeType, nodeData, app) {
//         const onNodeCreated = nodeType.prototype.onNodeCreated;
//         nodeType.prototype.onNodeCreated = function () {
//             onNodeCreated?.apply(this, arguments);
//             this.preview_images = [];
//         };

//         const onDrawBackground = nodeType.prototype.onDrawBackground;
//         nodeType.prototype.onDrawBackground = function (ctx) {
//             onDrawBackground?.apply(this, arguments);
//             if (this.preview_images && this.preview_images.length > 0) {
//                 const imgElement = ctx.querySelector("img");
//                 if (imgElement) {
//                     const imageData = this.preview_images[0];
//                     imgElement.src = `view?filename=${imageData.filename}&type=${imageData.type}&subfolder=${imageData.subfolder}`;
//                 }
//             }
//         };

//         const onExecuted = nodeType.prototype.onExecuted;
//         nodeType.prototype.onExecuted = function (message) {
//             onExecuted?.apply(this, arguments);
//             console.log(message);
//             if (message.images) {
//                 this.preview_images = message.images;
//                 console.log(message.images);
//                 this.setDirtyCanvas(true);
//             }
//         };
//     },
// });


app.registerExtension({
	name: "ComfyUI-Annotations-ImagePreview",

	nodeCreated(node, app) {
		if(node.comfyClass == "AnythingCache") {
			node._imgs = [new Image()];
			node.imageIndex = 0;

      let set_img_act = (v) => {
				node._img = v;
				var canvas = document.createElement('canvas');
				canvas.width = v[0].width;
				canvas.height = v[0].height;

				var context = canvas.getContext('2d');
				context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

				// var base64Image = canvas.toDataURL('image/png');
				// w.value = base64Image;
			};

      let already_tried = false;

			Object.defineProperty(node, 'imgs', {
				set(v) {
          console.log("Setting images");
          console.log(v);
					if(v && v.length == 0){
            console.log("No images!");
          	return;
          }
					node._imgs = v;
				},
				get() {
          if (!already_tried) {
            already_tried = true;
            try {
              if(node._imgs[0].src == '') {
                const subfolder = "ComfyUI-Annotations";
                const type = "temp";
                const filename = `preview-${node.id}.png`;

                let params = `?filename=${filename}&type=${type}&subfolder=${subfolder}`;

                api.fetchApi('/view/validate'+params, { cache: "no-store" }).then(response => {
                  if(response.status == 200) {
                    console.log("Got image!");      
                    node._imgs[0].src = 'view'+params;
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
