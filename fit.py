import torch.nn as nn
import torch.optim as optim
from play_game import play_single_game
from Model import TorchModel
from tqdm.auto import tqdm 
from ReplayBuffer import ReplayBuffer
import torch 
import numpy as np
from seed_all import seed_all


def eval_model(
    model: TorchModel,
    num_iter: int
):
    games_res = np.array(
        [
            play_single_game(model=model)
            for _ in range(num_iter)
        ],
        dtype=np.int32
    )

    return np.mean(games_res), np.sum(games_res == 2048)


def fit(  
    batch_size: int,
    epoch_num: int,
    games_per_epoch: int,
    batch_num: int,
    check_model_every_n_epochs: int, 
    gamma: float = 0.99
) -> None:
    seed_all()

    model = TorchModel(10)

    score_before, wins_before = eval_model(model=model, num_iter=1000)
    print(f"[epoch: {-1} | score: {score_before:.4f} | wins: {wins_before:.2f}]")


    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,        
        betas=(0.9, 0.999),  
        eps=1e-08,       
        weight_decay=0   
    )

    memory = ReplayBuffer(capacity=10 ** 6)
    loss_fn = nn.MSELoss()

    for epoch_num in tqdm(range(epoch_num)):
        model.eval()
        for _ in range(games_per_epoch):
            play_single_game(
                model=model,
                memory=memory
            )

        model.train()
        for batch_id in range(batch_num):
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            
            # Double DQN
            with torch.no_grad():
                next_preds = model(next_states)
                next_actions = next_preds.max(1)[1]
                next_q_values = next_preds.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + gamma * next_q_values * (~dones).float()
            
            current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            loss = loss_fn(current_q_values, target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        memory.clear()

        if (epoch_num + 1) % check_model_every_n_epochs == 0:
            model.eval()
            score, wins = eval_model(model=model, num_iter=1000)

            print(f"[epoch: {epoch_num + 1} | score: {score:.4f} | wins: {wins:.2f}]")

    score_after, wins_after = eval_model(model=model, num_iter=1000)
    print(f"[score: {score_after:.4f} | wins: {wins_after:.2f}]")


fit(  
    batch_size=1024,
    epoch_num=5,
    games_per_epoch=100,
    batch_num=5,
    check_model_every_n_epochs=2, 
)        
