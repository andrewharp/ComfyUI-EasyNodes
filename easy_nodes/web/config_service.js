// Passes config values from JavaScript up to Python via config_service.py
// Adapted from https://github.com/rgthree/rgthree-comfy
import { app } from '../../scripts/app.js'

export function getObjectValue(obj, objKey, def) {
    if (!obj || !objKey)
        return def;
    const keys = objKey.split(".");
    const key = keys.shift();
    const found = obj[key];
    if (keys.length) {
        return getObjectValue(found, keys.join("."), def);
    }
    return found;
}

export function setObjectValue(obj, objKey, value, createMissingObjects = true) {
    if (!obj || !objKey)
        return obj;
    const keys = objKey.split(".");
    const key = keys.shift();
    if (obj[key] === undefined) {
        if (!createMissingObjects) {
            return;
        }
        obj[key] = {};
    }
    if (!keys.length) {
        obj[key] = value;
    }
    else {
        if (typeof obj[key] != "object") {
            obj[key] = {};
        }
        setObjectValue(obj[key], keys.join("."), value, createMissingObjects);
    }
    return obj;
}

class EasyNodeApi {
    constructor(baseUrl) {
        this.baseUrl = baseUrl || './easy_nodes/api';
    }
    apiURL(route) {
        return `${this.baseUrl}${route}`;
    }
    fetchApi(route, options) {
        return fetch(this.apiURL(route), options);
    }
    async fetchJson(route, options) {
        const r = await this.fetchApi(route, options);
        return await r.json();
    }
}
export const easyNodeApi = new EasyNodeApi();

const easyNodesConfig = {};

class ConfigService extends EventTarget {
    getConfigValue(key, def) {
        return getObjectValue(easyNodesConfig, key, def);
    }
    async setConfigValues(changed) {
        const body = new FormData();
        body.append("json", JSON.stringify(changed));
        const response = await easyNodeApi.fetchJson("/config", { method: "POST", body });
        if (response.status === "ok") {
            for (const [key, value] of Object.entries(changed)) {
                setObjectValue(easyNodesConfig, key, value);
                this.dispatchEvent(new CustomEvent("config-change", { detail: { key, value } }));
            }
        }
        else {
            return false;
        }
        return true;
    }
}
export const SERVICE = new ConfigService();


export function createSetting(id, name, type, defaultValue, options) {
  app.ui.settings.addSetting({
    id,
    name,
    type,
    defaultValue,
    options,
    onChange: (value) => {
      SERVICE.setConfigValues({ [id]: value });
    },
  });
};
