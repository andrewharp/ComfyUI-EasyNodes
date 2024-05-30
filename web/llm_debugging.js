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

    createSetting(
      "easy_nodes.openai_api_token",
      "🧠 OpenAI Token (warning: this will be stored in plain text, so don't use on a shared system. In this case set the OPEN_AI_API_KEY environment variable instead)",
      "text",
      ""
    );
  },
});
