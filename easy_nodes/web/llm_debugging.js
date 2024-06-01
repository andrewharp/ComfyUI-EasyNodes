import { app } from "../../scripts/app.js";
import { createSetting } from "./config_service.js";

app.registerExtension({
  name: "easy_nodes.llm_debugging",
  async setup() {
    createSetting(
      "easy_nodes.llm_debugging",
      "🧠 LLM Debugging",
      "combo",
      "Off",
      (value) => [
        { value: "On", text: "On", selected: value === "On" },
        { value: "Off", text: "Off", selected: value === "Off" },
        { value: "AutoFix", text: "AutoFix", selected: value === "AutoFix" },
      ]
    );

    createSetting(
      "easy_nodes.max_tries",
      "🧠 LLM Max Tries",
      "number",
      3
    );

    createSetting(
      "easy_nodes.llm_model",
      "🧠 LLM Model",
      "text",
      "gpt-4o"
    );
  },
});
