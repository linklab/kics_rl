ENV_NAME = "TASKS_ALLOCATION"

STATIC_TASK_RESOURCE_DEMAND_SAMPLE = [
    [21, 21],
    [44, 25],
    [48, 32],
    [44, 47],
    [44, 49],
    [43, 53],
    [65, 64],
    [73, 69],
    [79, 89],
    [84, 98],
]

NUM_TASKS = 20

env_config = {
    "num_tasks": NUM_TASKS,  # 대기하는 태스크 개수
    "use_static_task_resource_demand": False,                           # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
    "use_same_task_resource_demand": False,                             # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
    "low_demand_resource_at_task": [1, 1],                              # 태스크의 각 자원 최소 요구량
    "high_demand_resource_at_task": [100, 100],                         # 태스크의 각 자원 최대 요구량
    "initial_resources_capacity": [NUM_TASKS * 30, NUM_TASKS * 30],     # 초기 자원 용량
}

if env_config["use_same_task_resource_demand"]:
    assert env_config["use_static_task_resource_demand"] is False

if env_config["use_static_task_resource_demand"]:
    assert env_config["use_same_task_resource_demand"] is False

if env_config["use_static_task_resource_demand"]:
    assert env_config["num_tasks"] == 10

dqn_config = {
    "max_num_episodes": 10000 * NUM_TASKS,              # 훈련을 위한 최대 에피소드 횟수
    "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
    "learning_rate": 0.001,                             # 학습율
    "gamma": 0.99,                                      # 감가율
    "steps_between_train": 4,                           # 훈련 사이의 환경 스텝 수
    "target_sync_step_interval": 500,                   # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
    "replay_buffer_size": 4000 * NUM_TASKS,             # 리플레이 버퍼 사이즈
    "epsilon_start": 0.95,                              # Epsilon 초기 값
    "epsilon_end": 0.01,                                # Epsilon 최종 값
    "epsilon_final_scheduled_percent": 0.75,            # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
    "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
    "train_num_episodes_before_next_test": 200,   # 검증 사이 마다 각 훈련 episode 간격
    "test_num_episodes": 30,                      # 검증에 수행하는 에피소드 횟수
    "early_stop_patience": NUM_TASKS,                   # episode_reward가 개선될 때까지 기다리는 기간
}

