"""
Process Control Unit (PIDManager) for Octivault AI Office.
Manages single-instance operation via PID file locking.
"""
import os
import logging
import signal
import errno
from typing import Union

logger = logging.getLogger("PIDManager")

class PIDManager:
    """
    The Process Control Unit of the Octivault AI Office.
    Manages the creation and cleanup of a PID file to ensure single instance
    operation and facilitate process management.
    """
    def __init__(self, pid_file_name: str = "octivault_trader.pid"):
        """
        Initialize PIDManager.
        
        Args:
            pid_file_name: Relative path from project root (e.g., 'logs/octivault_trader.pid')
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pid_file_path = os.path.join(project_root, pid_file_name)
        self._current_pid = os.getpid()
        self._is_locked_flag = False

        logger.info(f"💾 PIDManager (Process Control Unit) initialized. PID file will be: {self.pid_file_path}")

    def acquire_lock(self) -> bool:
        """Acquire process lock by creating PID file."""
        logger.debug(f"acquire_lock: Starting lock acquisition for PID {self._current_pid}.")
        logger.debug(f"acquire_lock: Checking PID file path: {self.pid_file_path}")

        # Ensure the directory for the PID file exists
        pid_dir = os.path.dirname(self.pid_file_path)
        if not os.path.exists(pid_dir):
            logger.debug(f"acquire_lock: PID file directory '{pid_dir}' does not exist. Attempting to create.")
            try:
                os.makedirs(pid_dir, exist_ok=True)
                logger.debug(f"acquire_lock: Created PID file directory '{pid_dir}'.")
            except Exception as e:
                logger.critical(f"acquire_lock: Failed to create PID file directory '{pid_dir}': {e}")
                return False

        if os.path.exists(self.pid_file_path):
            logger.debug(f"acquire_lock: PID file '{self.pid_file_path}' EXISTS.")
            try:
                with open(self.pid_file_path, 'r') as f:
                    existing_pid_str = f.read().strip()
                    existing_pid = int(existing_pid_str)
                logger.debug(f"acquire_lock: Read existing PID from file: '{existing_pid_str}', parsed as {existing_pid}")

                if self._is_pid_running(existing_pid):
                    logger.error(f"❌ Another instance of the bot is already running (PID: {existing_pid} from file '{self.pid_file_path}'). Exiting.")
                    return False
                else:
                    logger.warning(f"acquire_lock: Stale PID file found at '{self.pid_file_path}' for PID {existing_pid}. It appears the previous process is no longer running. Overwriting.")
            except (ValueError, IOError) as e:
                logger.warning(f"acquire_lock: Could not read or parse existing PID file '{self.pid_file_path}': {e}. Attempting to overwrite.")
        else:
            logger.debug(f"acquire_lock: PID file '{self.pid_file_path}' does NOT exist.")

        try:
            with open(self.pid_file_path, 'w') as f:
                f.write(str(self._current_pid))
            self._is_locked_flag = True
            logger.info(f"✅ PID file '{self.pid_file_path}' created successfully for current process (PID: {self._current_pid}). Lock acquired.")
            return True
        except IOError as e:
            logger.critical(f"acquire_lock: FAILED to create PID file at '{self.pid_file_path}': {e}", exc_info=True)
            self._is_locked_flag = False
            return False
        except Exception as e:
            logger.critical(f"acquire_lock: An unexpected error occurred during PID file creation: {e}", exc_info=True)
            self._is_locked_flag = False
            return False

    def remove_pid_file(self):
        """Remove the PID file."""
        logger.debug(f"remove_pid_file: Attempting to remove PID file: {self.pid_file_path}")
        if os.path.exists(self.pid_file_path):
            try:
                os.remove(self.pid_file_path)
                self._is_locked_flag = False
                logger.info(f"🗑️ PID file '{self.pid_file_path}' removed successfully. Lock released.")
            except OSError as e:
                logger.warning(f"remove_pid_file: Could not remove PID file '{self.pid_file_path}': {e}")
        else:
            logger.debug("remove_pid_file: No PID file found to remove.")

    def _is_pid_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        logger.debug(f"_is_pid_running: Checking status of PID {pid}")
        if pid <= 0:
            logger.debug(f"_is_pid_running: PID {pid} is invalid (<=0).")
            return False
        try:
            os.kill(pid, 0)
            logger.debug(f"_is_pid_running: PID {pid} is running.")
            return True
        except OSError as e:
            if e.errno == errno.ESRCH:
                logger.debug(f"_is_pid_running: PID {pid} does NOT exist (No such process).")
                return False
            elif e.errno == errno.EPERM:
                logger.warning(f"_is_pid_running: Permission denied to check PID {pid}. Assuming it's running.")
                return True
            else:
                logger.error(f"_is_pid_running: Error checking PID {pid}: {e}")
                return False
        except Exception as e:
            logger.error(f"_is_pid_running: Unexpected error checking PID {pid}: {e}", exc_info=True)
            return False

    def is_locked(self) -> bool:
        """Check if the lock is currently held by this process."""
        pid_in_file = self.get_pid_from_file()
        is_my_pid = (pid_in_file == self._current_pid)
        logger.debug(f"is_locked: _is_locked_flag={self._is_locked_flag}, PID in file={pid_in_file}, current_PID={self._current_pid}, match={is_my_pid}")
        return self._is_locked_flag and is_my_pid

    def get_pid(self) -> int:
        """
        Get the PID from the file if different from current, otherwise return current PID.
        Used to identify which process is running.
        """
        file_pid = self.get_pid_from_file()
        if file_pid is not None and file_pid != self._current_pid:
            return file_pid
        return self._current_pid

    def get_pid_from_file(self) -> Union[int, None]:
        """Read and return the PID from the PID file, or None if not found."""
        logger.debug(f"get_pid_from_file: Checking for file '{self.pid_file_path}'")
        if os.path.exists(self.pid_file_path):
            try:
                with open(self.pid_file_path, 'r') as f:
                    pid_str = f.read().strip()
                    if not pid_str:
                        logger.warning(f"get_pid_from_file: PID file '{self.pid_file_path}' is empty.")
                        return None
                    pid = int(pid_str)
                    logger.debug(f"get_pid_from_file: Read PID {pid} from '{self.pid_file_path}'.")
                    return pid
            except (ValueError, IOError) as e:
                logger.error(f"get_pid_from_file: Could not read or parse PID from file '{self.pid_file_path}': {e}")
                return None
        logger.debug(f"get_pid_from_file: PID file '{self.pid_file_path}' does not exist.")
        return None

__all__ = ['PIDManager']
