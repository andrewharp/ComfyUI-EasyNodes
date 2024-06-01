import { app } from "../../scripts/app.js";
import { createSetting } from "./config_service.js";

const retainPreviewsId = "easy_nodes.RetainPreviews";


app.registerExtension({
    name: "Retain Previews",

    async setup() {
        createSetting(
            retainPreviewsId,
            "🪄 Save preview images across browser sessions. Requires initial refresh to activate/deactivate.",
            "boolean",
            false,
        );
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!app.ui.settings.getSettingValue(retainPreviewsId)) {
            return;
        }

        const previewTypes = [
            "PreviewImage", "PreviewMask", "PreviewDepth", "PreviewNormal",
             "AnythingCache", "PlotLosses", "SaveAnimatedPNG", "SaveImage"];

        if (previewTypes.includes(nodeData.name)) {
            console.log("Found preview node: " + nodeData.name);
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (output) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [output]);
                this.canvasWidget.value = output;
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated === null || onNodeCreated === void 0 ? void 0 : onNodeCreated.apply(this);

                const node = this;
                const widget = {
                    type: "dict",
                    name: "Retain_Previews",
                    options: { serialize: false },
                    _value: {},
                    set value(v) {
                        this._value = v;
                        app.nodeOutputs[node.id + ""] = v;
                    },

                    get value() {
                        return this._value;
                    },
                };
                
                this.canvasWidget = this.addCustomWidget(widget);
            }
        }
    },
});
