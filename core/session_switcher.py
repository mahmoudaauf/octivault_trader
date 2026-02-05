import logging
from datetime import datetime, time
import pytz

class SessionSwitcher:
    def __init__(self, agent_manager, config):
        self.agent_manager = agent_manager
        self.config = config
        self.logger = logging.getLogger("SessionSwitcher")

        # Define session time windows (UTC)
        self.sessions = {
            "Asia": (time(0, 0), time(8, 0)),
            "EU": (time(8, 0), time(16, 0)),
            "US": (time(16, 0), time(23, 59)),
        }

        # Load schedule: agent_name -> list of active sessions
        self.agent_schedule = self.config.AGENT_SESSION_MAP  # e.g. {"DipSniper": ["US", "EU"], "IPOChaser": ["Asia"]}

    def get_current_session(self):
        now_utc = datetime.utcnow().time()
        for session, (start, end) in self.sessions.items():
            if start <= now_utc <= end:
                return session
        return None

    def update_agents(self):
        current_session = self.get_current_session()
        self.logger.info(f"🕒 Current session: {current_session}")

        for agent_name, agent in self.agent_manager.agent_lookup.items():
            allowed_sessions = self.agent_schedule.get(agent_name, [])
            if current_session in allowed_sessions:
                if not self.agent_manager.is_active(agent_name):
                    self.logger.info(f"✅ Activating agent: {agent_name} for session {current_session}")
                    self.agent_manager.activate(agent_name)
            else:
                if self.agent_manager.is_active(agent_name):
                    self.logger.info(f"⛔ Deactivating agent: {agent_name} (not in session {current_session})")
                    self.agent_manager.deactivate(agent_name)
