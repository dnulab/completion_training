# ChatGPT Export: L2-SP regularization for model adaptation / fine-tuning in PyTorch

Exported July 31, 2026. User is John MacCormick. Model is GPT5.6 Sol (light reasoning)

## User

Suppose I am interested in doing model adaptation or fine-tuning by taking a pre-trained GPT model and giving it new data. I want to prevent it forgetting the data it's already learned. I want to use a simple approach that adds a penalty to the loss function. The penalty is proportional to the L2 norm between current weights and the old weights before the fine-tuning began. Can you give me a high-level overview of how to implement this in PyTorch?

## Assistant

This approach is often called **L2-SP regularization**: fine-tune normally, but discourage parameters from moving too far from their pretrained values.

The objective is typically:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{task}
+
\lambda \sum_i \|\theta_i-\theta_i^{old}\|_2^2
$$

Here, $\theta^{old}$ is a frozen snapshot of the pretrained weights and $\lambda$ controls the stability–adaptation tradeoff.

```python
import torch
import torch.nn.functional as F

model = load_pretrained_model()
model.train()

# Store fixed copies of the parameters before fine-tuning.
old_params = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
l2sp_strength = 1e-4

for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )

    # Example causal-language-model loss.
    task_loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(-1, outputs.logits.size(-1)),
        batch["input_ids"][:, 1:].reshape(-1),
    )

    penalty = torch.zeros((), device=task_loss.device)

    for name, param in model.named_parameters():
        if name in old_params:
            reference = old_params[name].to(param.device)
            penalty = penalty + (param - reference).pow(2).sum()

    loss = task_loss + l2sp_strength * penalty
    loss.backward()
    optimizer.step()
```

A few practical details matter:

- **Keep the old weights constant.** Use `detach().clone()` and never give them to the optimizer.
- **Usually use squared L2 distance.** The literal L2 norm would be `torch.linalg.vector_norm(...)`, but its square is smoother and more commonly used.
- **Normalize if needed.** Because the summed penalty grows with model size, you can divide by the number of penalized parameters:

  ```python
  penalty = penalty / sum(p.numel() for p in old_params.values())
  ```

- **Tune $\lambda$.** A large value preserves pretrained behavior but may prevent useful adaptation; a small value allows more learning but offers less protection.
- **Consider parameter groups.** You might regularize transformer layers strongly while allowing a newly initialized output head to move freely.
- **Watch memory usage.** Saving the reference parameters requires roughly another model-sized block of memory. They can be stored on CPU and transferred layer-by-layer, at some performance cost.
- **Be mindful of AdamW.** Its `weight_decay` penalizes weights relative to zero, whereas this penalty anchors them to their pretrained values. They are different mechanisms.

This reduces parameter drift, but it does not guarantee that old capabilities or facts will be retained. Mixing some original-domain data into fine-tuning—rehearsal—is usually stronger when such data is available. More sophisticated variants, such as elastic weight consolidation, weight each parameter according to its estimated importance.

