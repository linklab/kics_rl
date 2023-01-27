# env_config = {
#     "num_tasks": 10,  # 대기하는 태스크 개수
#     "use_static_task_resource_demand": True,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
#     "same_task_resource_demand": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
#     "initial_resources_capacity": [100, 100],  # 초기 자원 용량
#     "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
#     "high_demand_resource_at_task": [20, 20]  # 태스크의 각 자원 최대 요구량
# }

env_config = {
    "num_tasks": 3,  # 대기하는 태스크 개수
    "use_static_task_resource_demand": False,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
    "same_task_resource_demand": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
    "initial_resources_capacity": [30, 30],  # 초기 자원 용량
    "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
    "high_demand_resource_at_task": [20, 20]  # 태스크의 각 자원 최대 요구량
}

if env_config["same_task_resource_demand"]:
    assert env_config["use_static_task_resource_demand"] is False

if env_config["use_static_task_resource_demand"]:
    assert env_config["num_tasks"] == 10

dqn_config = {
    "max_num_episodes": 20_000,  # 훈련을 위한 최대 에피소드 횟수
    "batch_size": 4,  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
    "learning_rate": 0.0001,  # 학습율
    "gamma": 0.99,  # 감가율
    "use_action_mask": True,
    "target_sync_step_interval": 500,  # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
    "replay_buffer_size": 300_000,  # 리플레이 버퍼 사이즈
    "epsilon_start": 0.95,  # Epsilon 초기 값
    "epsilon_end": 0.01,  # Epsilon 최종 값
    "epsilon_final_scheduled_percent": 0.75,  # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
    "print_episode_interval": 10,  # Episode 통계 출력에 관한 에피소드 간격
    "train_num_episodes_before_next_validation": 200,  # 검증 사이 마다 각 훈련 episode 간격
    "validation_num_episodes": 30,  # 검증에 수행하는 에피소드 횟수
    "early_stop_patience": env_config["num_tasks"] * 10_000,  # episode_reward가 개선될 때까지 기다리는 기간
}
