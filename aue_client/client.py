# aue client 
# the rl decision sender posts ratio updates to the aue (post /ratio).

import time
from typing import Dict, Any, Optional
import requests
import logging
from dataclasses import dataclass
from threading import Event
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@dataclass
class RatioDecision:
    wifi_ratio: float
    fiveg_ratio: float
    timestamp_us: int
    session_id: str


class AUEClient:
    def __init__(self, aue_url: str, timeout: float = 5.0):
        self.aue_url = aue_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True,
    )
    def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        return self.session.post(url, json=payload, timeout=self.timeout)

    def get_timestamp(self) -> str:
        try:
            url = f"{self.aue_url}/rl-timestamp"
            response = self.session.get(url)

            if response.status_code == 200:
                return response.json().get("timestamp")
            else:
                logger.warning(
                    f"AUE returned status {response.status_code} for {url}: "
                    f"{response.text}"
                )
                return f'ERR_STATUS_{response.status_code}'

        except requests.RequestException as e:
            logger.error(f"AUE request failed: {e}")
            return 'ERR_REQUEST_FAILED'

    def send_ratio(self, wifi_ratio: float) -> bool:
        if not isinstance(wifi_ratio, (int, float)):
            logger.error(f"Invalid wifi_ratio type: {type(wifi_ratio)}")
            return False
        if not (0.0 <= wifi_ratio <= 1.0):
            logger.error(f"Invalid wifi_ratio: {wifi_ratio} (must be 0.0-1.0)")
            return False

        try:
            ratio_wifi = round(wifi_ratio * 100)
            ratio_wifi = max(0, min(100, ratio_wifi))

            # TODO: Hotfix, it would be better if this method took 5g_ratio instead of wifi_ratio
            ratio_5g = 100 - ratio_wifi

            url = f"{self.aue_url}/ratio"
            payload = {"ratio": ratio_5g}

            response = self._post_with_retry(url, payload)

            if response.status_code == 200:
                logger.info(f"Sent ratio to AUE: {ratio_wifi}% WiFi ({ratio_5g}% 5G)")
                return True
            else:
                logger.warning(
                    f"AUE returned status {response.status_code} for {url}: "
                    f"{response.text[:200]}"
                )
                return False

        except requests.RequestException as e:
            logger.error(f"AUE request failed: {e}")
            return False
