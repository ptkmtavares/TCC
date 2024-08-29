import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    
def save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(G, D, optimizer_G, optimizer_D, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    return checkpoint['epoch']

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def train_gan(G, D, train_loader, input_dim, num_epochs=1000, n_critic=1, device='cpu', checkpoint_dir='checkpoints'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(D.parameters(), lr=0.00005)
    grad_scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"🔄 Loading latest checkpoint: {latest_checkpoint}")
        start_epoch = load_checkpoint(G, D, optimizer_G, optimizer_D, latest_checkpoint)
        if start_epoch >= num_epochs:
            print(
                f"🔙 Checkpoint epoch is greater than or equal to num_epochs. Returning...\n"
                f"{'='*75}"
            )
            return
        print(
            f"✅ Resuming training from epoch {start_epoch}\n"
            f"{'='*75}"
        )
    save_interval = (num_epochs - start_epoch) // 4
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        for real_data, _ in train_loader:
            real_data = real_data.to(device)
            real_labels = torch.ones(real_data.size(0), 1).to(device) * 0.9
            fake_labels = torch.zeros(real_data.size(0), 1).to(device) + 0.1

            for _ in range(n_critic):
                noise = torch.randn(real_data.size(0), input_dim).to(device)
                fake_data = G(noise)
                optimizer_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    real_loss = criterion(D(real_data), real_labels)
                    fake_loss = criterion(D(fake_data.detach()), fake_labels)
                    d_loss = real_loss + fake_loss
                grad_scaler.scale(d_loss).backward()
                grad_scaler.step(optimizer_D)
                grad_scaler.update()

            noise = torch.randn(real_data.size(0), input_dim).to(device)
            fake_data = G(noise)
            optimizer_G.zero_grad()
            with torch.amp.autocast('cuda'):
                g_loss = criterion(D(fake_data), real_labels)
            grad_scaler.scale(g_loss).backward()
            grad_scaler.step(optimizer_G)
            grad_scaler.update()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = (num_epochs - (epoch + 1)) * epoch_duration

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            print(
                f"🌟 Epoch [{epoch + 1}/{num_epochs}]\n"
                f"🥊 D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}\n"
                f"⏳ Time for this epoch: {epoch_duration:.2f} seconds\n"
                f"⏱️ Estimated remaining time: {remaining_time:.2f} seconds\n"
                f"{'='*75}"
            )
            save_checkpoint(G, D, optimizer_G, optimizer_D, epoch + 1, checkpoint_dir)

def generate_adversarial_examples(G, num_samples, input_dim, device='cpu'):
    noise = torch.randn(num_samples, input_dim).to(device)
    generated_data = G(noise).cpu().detach().numpy()
    return generated_data