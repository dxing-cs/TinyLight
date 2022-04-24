from .base_agent import BaseAgent

from agent.drl_based.base_drl import BaseDRL
from agent.drl_based.colight import CoLight
from agent.drl_based.ecolight import EcoLight
from agent.drl_based.frap import FRAP
from agent.drl_based.mplight import MPLight

from agent.rule_based.fixed_time import FixedTime
from agent.rule_based.max_pressure import MaxPressure
from agent.rule_based.sotl import SOTL

from agent.tiny_light.tiny_light import TinyLight
from agent.tiny_light.tiny_light_quan import TinyLightQuan
from agent.tiny_light.random_path import RandomPath
