import os
import json
import re
from server import PromptServer
from aiohttp import web

# Receives config values from config_service.js
# Code in this file adapted from https://github.com/rgthree/rgthree-comfy

def get_dict_value(data: dict, dict_key: str, default = None):
  """ Gets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  found = data[key] if key in data else None
  if found is not None and len(keys) > 0:
    return get_dict_value(found, '.'.join(keys), default)
  return found if found is not None else default


def set_dict_value(data: dict, dict_key: str, value, create_missing_objects = True):
  """ Sets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key not in data:
    if not create_missing_objects:
      return
    data[key] = {}
  if len(keys) == 0:
    data[key] = value
  else:
    set_dict_value(data[key], '.'.join(keys), value, create_missing_objects)

  return data


def dict_has_key(data: dict, dict_key):
  """ Checks if a dict has a deeply nested dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key is None or key not in data:
    return False
  if len(keys) == 0:
    return True
  return dict_has_key(data[key], '.'.join(keys))


def get_config_value(key, default = None):
    # logging.info(f"Getting config value for key: {key} from {USER_CONFIG}")
    return get_dict_value(USER_CONFIG, key, default)
  

def extend_config(default_config, user_config):
  """ Returns a new config dict combining user_config into defined keys for default_config."""
  cfg = {}
  for key, value in default_config.items():
    if key not in user_config:
      cfg[key] = value
    elif isinstance(value, dict):
      cfg[key] = extend_config(value, user_config[key])
    else:
      cfg[key] = user_config[key] if key in user_config else value
  return cfg


def set_user_config(data: dict):
  """ Sets the user configuration."""
  count = 0
  for key, value in data.items():
    # if dict_has_key(USER_CONFIG, key):
    set_dict_value(USER_CONFIG, key, value, True)
    count+=1
  if count > 0:
    write_user_config()


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(THIS_DIR, '..', 'easy_node_config.json.default')
USER_CONFIG_FILE = os.path.join(THIS_DIR, '..', 'easy_node_config.json')


def get_user_config():
  """ Gets the user configuration."""
  if os.path.exists(USER_CONFIG_FILE):
    with open(USER_CONFIG_FILE, 'r', encoding = 'UTF-8') as file:
      config = re.sub(r"(?:^|\s)//.*", "", file.read(), flags=re.MULTILINE)
    return json.loads(config)
  else:
    return {}


def write_user_config():
  """ Writes the user configuration."""
  with open(USER_CONFIG_FILE, 'w+', encoding = 'UTF-8') as file:
    json.dump(USER_CONFIG, file, sort_keys=True, indent=2, separators=(",", ": "))


USER_CONFIG = get_user_config()


# Migrate old config options into "features"
needs_to_write_user_config = False


if needs_to_write_user_config is True:
  print('writing new user config.')
  write_user_config()


routes = PromptServer.instance.routes


@routes.get('/easy_nodes/config.js')
def api_get_user_config_file(request):
  """ Returns the user configuration as a javascript file. """
  text=f'export const rgthreeConfig = {json.dumps(USER_CONFIG, sort_keys=True, indent=2, separators=(",", ": "))}'
  return web.Response(text=text, content_type='application/javascript')


@routes.get('/easy_nodes/api/config')
def api_get_user_config(request):
  """ Returns the user configuration. """
  return web.json_response(json.dumps(USER_CONFIG))


@routes.post('/easy_nodes/api/config')
async def api_set_user_config(request):
  """ Returns the user configuration. """
  post = await request.post()
  data = json.loads(post.get("json"))
  set_user_config(data)
  return web.json_response({"status": "ok"})
