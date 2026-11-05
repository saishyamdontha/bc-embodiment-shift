import gymnasium as gym

# Source robot: default dynamics. Target robot: strong wind and turbulence.
DOMAINS = {
    "source": {},
    "target": dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5),
}
SUCCESS_RETURN = 200.0   # standard "solved" threshold for LunarLander


def make(domain, **kw):
    return gym.make("LunarLander-v3", **DOMAINS[domain], **kw)
