import torch
import torch.nn.functional as F

def integrated_gradients(
    model,
    tokenizer,
    content,
    target_token_id=None,
    steps=50,
    device="cuda"
):
    model.eval()

    prompt = [{"role": "user", "content": content},]

    # 1. Tokenize
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        # return_offsets_mapping=True,
        return_tensors="pt",
    ).to(device)
    input_ids = inputs["input_ids"]

    # Tokenize user content alone (no chat template)
    user_ids = tokenizer(
        content,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0].to(device)

    # Find where user tokens appear in full input
    full_ids = input_ids[0]

    # Simple subsequence match
    def find_subsequence(sequence, subseq):
        for i in range(len(sequence) - len(subseq) + 1):
            if torch.equal(sequence[i:i+len(subseq)], subseq):
                return i, i + len(subseq)
        return None, None

    user_start, user_end = find_subsequence(full_ids, user_ids)

    # 2. Get embeddings
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)

    # 3. Baseline (PAD tokens)
    baseline_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    baseline_ids = torch.full_like(input_ids, baseline_token_id)
    baseline_embeds = embed_layer(baseline_ids)

    # 4. Choose target token if not given
    if target_token_id is None:
        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeds)
            target_token_id = outputs.logits[0, -1].argmax().item()

    # 5. Interpolation coefficients
    alphas = torch.linspace(0, 1, steps).to(device)

    total_gradients = torch.zeros_like(input_embeds)

    # 6. Integrated gradient loop
    for alpha in alphas:
        interpolated_embeds = (
            baseline_embeds + alpha * (input_embeds - baseline_embeds)
        )
        interpolated_embeds.requires_grad_(True)
        interpolated_embeds.retain_grad()

        outputs = model(inputs_embeds=interpolated_embeds)
        target_logit = outputs.logits[0, -1, target_token_id]

        grads = torch.autograd.grad(
            outputs=target_logit,
            inputs=interpolated_embeds,
            retain_graph=False,
            create_graph=False
        )[0]

        total_gradients += grads.detach()

    # 7. Average gradients
    avg_gradients = total_gradients / steps

    # 8. Integrated gradients
    integrated_grads = (input_embeds - baseline_embeds) * avg_gradients

    # 9. Token-level attribution
    token_attributions = integrated_grads.sum(dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attributions = token_attributions.detach().cpu()

    if user_start is not None:
        tokens = tokens[user_start:user_end]
        attributions = attributions[user_start:user_end]

    return {
        "tokens": tokens,
        "attributions": attributions
    }
