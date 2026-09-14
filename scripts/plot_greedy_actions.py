import matplotlib.pyplot as plt
import numpy as np


def get_rewards(wifi_ratio, wifi_rtt, fiveg_rtt, wifi_plr, fiveg_plr, epsilon=1e-6, r_max=1000):
    wifi_ratio = np.clip(wifi_ratio, a_min=0, a_max=1)
    weighted_rtt = wifi_ratio * wifi_rtt + (1 - wifi_ratio) * fiveg_rtt
    weighted_plr = wifi_ratio * wifi_plr + (1 - wifi_ratio) * fiveg_plr

    rtt_norm = np.clip(weighted_rtt / r_max, a_min=0, a_max=1)
    plr_norm = np.clip(weighted_plr, a_min=0, a_max=1.0)

    rtt_term = np.log(1.0 - rtt_norm + epsilon)
    plr_term = np.log(1.0 - plr_norm + epsilon)
    return rtt_term + plr_term


def main():
    all_actions = np.arange(0, 1.01, 0.01)

    all_rtt = np.arange(0, 50, 5)
    all_plr = np.arange(0, 0.05, 0.005)

    max_rewards = np.zeros((len(all_rtt), len(all_rtt), len(all_plr), len(all_plr)), dtype=np.float16)
    optimal_actions = np.zeros((len(all_rtt), len(all_rtt), len(all_plr), len(all_plr)), dtype=np.float16)

    for wifi_rtt_i, wifi_rtt in enumerate(all_rtt):
        for fiveg_rtt_i, fiveg_rtt in enumerate(all_rtt):
            for wifi_plr_i, wifi_plr in enumerate(all_plr):
                for fiveg_plr_i, fiveg_plr in enumerate(all_plr):
                    all_rewards = get_rewards(all_actions, wifi_rtt, fiveg_rtt, wifi_plr, fiveg_plr)
                    argmax_idx = np.argmax(all_rewards)
                    argmax_is_same_as_last_action = all_rewards[argmax_idx] == all_rewards[-1]
                    if argmax_is_same_as_last_action:
                        argmax_idx = -1
                    max_rewards[wifi_rtt_i, fiveg_rtt_i, wifi_plr_i, fiveg_plr_i] = all_rewards[argmax_idx]
                    optimal_actions[wifi_rtt_i, fiveg_rtt_i, wifi_plr_i, fiveg_plr_i] = all_actions[argmax_idx]

    mixed_actions = ((optimal_actions > 0) * (optimal_actions < 1)).sum()
    pure_actions = optimal_actions.size - mixed_actions
    print("Pure:", pure_actions, ", mixed", mixed_actions)

    # plt.imshow(optimal_actions[:, :, 0, 0], origin="lower")
    # plt.colorbar()
    # plt.xlabel("Wifi RTT")
    # plt.ylabel("5G RTT")
    # plt.show()

    plt.imshow(optimal_actions[:, 2, :, 3], origin="lower")
    plt.colorbar()
    plt.xlabel("Wifi RTT")
    plt.ylabel("Wifi PLR")
    plt.show()


if __name__ == "__main__":
    main()