from pmf_client.client import PMFClientManager
import matplotlib.pyplot as plt
import numpy as np


def main():
    episode_length = 100
    num_timesteps = 2000
    assert num_timesteps % episode_length == 0

    pmf_client = PMFClientManager(dry_run=True, pmf_ue_url="", pmf_upf_url="")

    wifi_rtt, fiveg_rtt, wifi_plr, fiveg_plr = np.zeros(num_timesteps), np.zeros(num_timesteps), np.zeros(num_timesteps), np.zeros(num_timesteps)
    for t in range(0, num_timesteps):
        if t % episode_length == 0:
            pmf_client.episode_reset_callback()

        wifi_rtt[t], fiveg_rtt[t], wifi_plr[t], fiveg_plr[t] = pmf_client.get_dry_run_metrics_mean()

    x = np.arange(num_timesteps)

    def insert_episode_breaks(arr):
        arr_l = []
        for t in range(0, num_timesteps):
            if t % episode_length == 0:
                arr_l.append(np.nan)
            arr_l.append(arr[t])
        return np.array(arr_l)

    x = insert_episode_breaks(x)
    wifi_rtt = insert_episode_breaks(wifi_rtt)
    wifi_plr = insert_episode_breaks(wifi_plr)
    fiveg_rtt = insert_episode_breaks(fiveg_rtt)
    fiveg_plr = insert_episode_breaks(fiveg_plr)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1 = plt.plot(x, wifi_rtt, label="Wifi RTT")
    l2 = plt.plot(x, fiveg_rtt, label="5G RTT")

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Round trip time (ms)")
    ax1.grid(True)

    ax2 = ax1.twinx()

    # advance color two times
    ax2.plot([], [])
    ax2.plot([], [])
    l3 = ax2.plot(x, wifi_plr, label="WiFi PLR", linestyle="--")
    l4 = ax2.plot(x, fiveg_plr, label="5G PLR", linestyle="--")

    ax2.set_ylabel("Packet Loss Ratio")

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
