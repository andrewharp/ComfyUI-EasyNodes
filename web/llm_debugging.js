import { app } from "../../../scripts/app.js";
import { SERVICE } from "./easynode_config_service.js";

const llmDebugingId = "easy_nodes.llm_debugging";
const maxTriesId = "easy_nodes.max_tries";
const llmModelId = "easy_nodes.llm_model";
const openAiTokenId = "easy_nodes.openai_api_token";

const ext = {
  name: llmDebugingId,
  async setup(app) {
    app.ui.settings.addSetting({
      id: llmDebugingId,
      name: "🧠 LLM Debugging",
      defaultValue: "Off",
      type: "combo",
      options: (value) => [
        { value: "On", text: "On", selected: value === "On" },
        { value: "Off", text: "Off", selected: value === "Off" },
        { value: "AutoFix", text: "AutoFix", selected: value === "AutoFix" },
      ],
      onChange(value) {
        console.log("LLM Debugging: " + llmDebugingId + " " + value);
        SERVICE.setConfigValues({ [llmDebugingId]: value });
      },
    });

    app.ui.settings.addSetting({
      id: maxTriesId,
      name: "🧠 LLM Max Tries",
      defaultValue: 3,
      type: "number",
      onChange(value) {
        console.log("Max Tries: " + maxTriesId + " " + value);
        SERVICE.setConfigValues({ [maxTriesId]: value });
      },
    });

    app.ui.settings.addSetting({
      id: llmModelId,
      name: "🧠 LLM Model",
      defaultValue: "gpt-4o",
      type: "text",
      onChange(value) {
        SERVICE.setConfigValues({ [llmModelId]: value });
      },
    });

    app.ui.settings.addSetting({
      id: openAiTokenId,
      name: "🧠 OpenAI Token (warning: this will be stored in plain text, so don't use on a shared system. In this case set the OPEN_AI_API_KEY environment variable instead)",
      defaultValue: "",
      type: "text",
      onChange(value) {
        SERVICE.setConfigValues({ [openAiTokenId]: value });
      },
    });
  },
};

app.registerExtension(ext);
