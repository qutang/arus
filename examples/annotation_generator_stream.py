from arus.core.annotation.generator import normal_dist

if __name__ == "__main__":
    duration_mu = 5
    duration_sigma = 1
    start_time = None
    num_mu = 3
    labels = ['Sitting', 'Standing', 'Lying']
    sleep_interval = 0
    
    n = 5
    for data in normal_dist(duration_mu=duration_mu, duration_sigma=duration_sigma, start_time=start_time, num_mu=num_mu, labels=labels, sleep_interval=sleep_interval):
        print(data)
        n -= 1
        if n == 0:
            break