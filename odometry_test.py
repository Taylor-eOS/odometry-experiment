import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

grid_size = 5
img_size = 8
embedding_dim = img_size * img_size
actions = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}

def generate_grid_images(folder):
    os.makedirs(folder, exist_ok=True)
    for x in range(grid_size):
        for y in range(grid_size):
            seed = x * grid_size + y
            rng = np.random.RandomState(seed)
            arr = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(os.path.join(folder, f"{x}_{y}.png"))

def load_embeddings(folder):
    embeddings = {}
    for x in range(grid_size):
        for y in range(grid_size):
            path = os.path.join(folder, f"{x}_{y}.png")
            img = Image.open(path).convert('L').resize((img_size, img_size))
            v = torch.tensor(np.array(img), dtype=torch.float32).flatten()
            embeddings[(x, y)] = F.normalize(v, dim=0)
    return embeddings

def move(pos, action):
    dx, dy = actions[action]
    nx, ny = pos[0] + dx, pos[1] + dy
    if 0 <= nx < grid_size and 0 <= ny < grid_size:
        return (nx, ny)
    return pos

def one_hot_action(action):
    vec = torch.zeros(len(actions))
    vec[list(actions.keys()).index(action)] = 1
    return vec

def train_transition_model(world_embeddings):
    data = []
    for _ in range(2000):
        x, y = random.randrange(grid_size), random.randrange(grid_size)
        pos = (x, y)
        a = random.choice(list(actions.keys()))
        nxt = move(pos, a)
        src_emb = world_embeddings[pos]
        dst_emb = world_embeddings[nxt]
        data.append((torch.cat([src_emb, one_hot_action(a)]), dst_emb))
    X = torch.stack([d[0] for d in data])
    Y = torch.stack([d[1] for d in data])
    model = nn.Sequential(
        nn.Linear(embedding_dim + len(actions), 128),
        nn.ReLU(),
        nn.Linear(128, embedding_dim))
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(300):
        pred = model(X)
        loss = F.mse_loss(F.normalize(pred, dim=1), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def simulate_path(model, world_embeddings, start, path):
    pos = start
    emb = world_embeddings[pos]
    for a in path:
        inp = torch.cat([emb, one_hot_action(a)])
        emb = model(inp).flatten()
        emb = F.normalize(emb, dim=0)
        pos = move(pos, a)
        true_emb = world_embeddings[pos]
        sim = F.cosine_similarity(emb, true_emb, dim=0).item()
        print(f"action {a:>1} -> moved to {pos}, similarity {sim:.3f}")

if __name__ == "__main__":
    folder = "grid_images"
    generate_grid_images(folder)
    world_embeddings = load_embeddings(folder)
    model = train_transition_model(world_embeddings)
    simulate_path(model, world_embeddings, (2, 2), ['E', 'E', 'S', 'S', 'W'])

