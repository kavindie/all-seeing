import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import itertools
import numpy as np
import pulp
from scipy.optimize import minimize
import cvxpy as cp
from itertools import combinations
from matplotlib import pyplot as plt

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        line["text"] = "How many boxes are there?"
        qs = line["text"]
        if self.model_config.mm_use_im_start_end and self.model_config.mm_use_im_patch_token:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.model_config.num_query_tokens + DEFAULT_IM_END_TOKEN + '\n' + qs
        elif self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "a")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    frame_embeddings = []
    answers = []

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            input_ids_preped, position_ids,attention_mask,past_key_values,inputs_embeds,labels= model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            )
            outputs = model.model(  input_ids=input_ids_preped,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    use_cache=True,
                                    output_attentions=model.model.config.output_attentions,
                                    output_hidden_states=model.model.config.output_hidden_states,
                                    return_dict=model.model.config.use_return_dict,)
            last_hidden_state = outputs.last_hidden_state
            last_embedding = last_hidden_state[:, -1, :]
            frame_embeddings.append(last_embedding)

            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs_text = outputs_text.strip()
            answers.append(outputs_text)
    
    torch.save(torch.stack(frame_embeddings),'frame_embeddings_boxes.pt')
    with open('answers_boxes.pt', "w") as f:
        for string in answers:
            f.write(string + "\n") 

            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     max_new_tokens=args.max_new_tokens,
            #     use_cache=True)

        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = outputs.strip()

        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
    # ans_file.close()

def generate_pdf(filepath=None):
    if filepath is None:
        filepath = 'frame_embeddings.pt'
    frame_embeddings = torch.load(filepath)[:,0,:]
    frame_embeddings_norm = torch.nn.functional.normalize(frame_embeddings, dim=-1)
    
    video_embedding = frame_embeddings.mean(dim=0)
    video_embedding_norm = torch.nn.functional.normalize(video_embedding, dim=-1)

    cosine_similarity = torch.nn.functional.cosine_similarity(video_embedding_norm, frame_embeddings_norm)
    softmax_op = torch.nn.Softmax(dim=0)
    pdf = softmax_op(cosine_similarity)

    argmax_img_num = pdf.argmax().item() - 1
    argmax_img_path = f"/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/images/_xMr-HKMfVA/frame_{argmax_img_num:04d}.jpg"
    Image.open(argmax_img_path).save(f'/scratch3/kat049/all-seeing/all-seeing-v2/temp_sgc/argmax_img.jpg')

    best_x, best_error, best_lambda = solve_with_lasso(frame_embeddings_norm.to('cpu').T, video_embedding_norm.to('cpu'))
    visualize(np.nonzero(best_x)[0])

    video_data = user_annotations()
    video_id = '_xMr-HKMfVA'
    user_summary = video_data[video_id].frame_mean

    # downsample_num = 240
    # frame_embeddings_norm_subset = frame_embeddings_norm[::downsample_num]

    # selected_combo =  l0_optimization(frame_embeddings_norm_subset.to('cpu').T, video_embedding_norm.to('cpu'))
    # selected_indices_in_original = (np.array(selected_combo).nonzero()[0] * downsample_num).tolist()
    # l0_simple_img_num = selected_indices_in_original[0] - 1 # need to iterate for other combos
    # l0_simple_img_path = f"/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/images/_xMr-HKMfVA/frame_{l0_simple_img_num:04d}.jpg"
    # Image.open(l0_simple_img_path).save(f'/scratch3/kat049/all-seeing/all-seeing-v2/temp_sgc/l0_img.jpg')

    # array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def l0_optimization(A, y):
    """
    A: [m, n] = frame_embeddings_norm
    y: [m] = video_embedding_norm
    """
    number_of_frames = A.shape[1]
    binary_vectors = np.array(list(itertools.product([0, 1], repeat=number_of_frames)))

    selected_frame_combo = None
    lowest_l0_error = float('inf')

    for binary_vector in binary_vectors:
        Ax_norm = torch.nn.functional.normalize((A @ binary_vector), dim=0)

        l0_error = np.linalg.norm(Ax_norm - y)
        if l0_error < lowest_l0_error:
            selected_frame_combo = binary_vector
            lowest_l0_error = l0_error

    return selected_frame_combo   

def solve_with_lasso(A, y, alpha_range=None):
    """
    Use Lasso regression with automatic alpha tuning to find an approximate sparse solution.
    
    Parameters:
    A: np.ndarray of shape [m, n] - normalized frame embeddings
    y: np.ndarray of shape [m] - normalized video embedding
    alpha_range: list - custom range of alpha values to try
    
    Returns:
    x: np.ndarray - binary solution vector
    error: float - cosine distance between λAx and y
    lambda_scale: float - scaling factor λ
    """
    from sklearn.linear_model import Lasso
    if alpha_range is None:
        # Try a wide range of alpha values, exponentially spaced
        alpha_range = np.logspace(-4, 0, 20)
    
    best_x = None
    best_error = float('inf')
    best_lambda = None
    
    for alpha in alpha_range:
        # Solve using Lasso regression
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
        lasso.fit(A, y)
        
        # Skip if all coefficients are zero
        if np.all(lasso.coef_ == 0):
            continue
            
        # Try different thresholds for binarization
        percentiles = [50, 60, 70, 80, 90]
        for p in percentiles:
            threshold = np.percentile(lasso.coef_[lasso.coef_ > 0], p) if np.any(lasso.coef_ > 0) else 0
            x = np.where(lasso.coef_ > threshold, 1, 0)
            
            # Skip if all zeros
            if np.sum(x) == 0:
                continue
                
            # Calculate error and scaling factor
            Ax = A @ x
            lambda_scale = 1.0 / (np.linalg.norm(Ax) + 1e-10)
            Ax_normalized = Ax * lambda_scale
            error = 1.0 - np.dot(Ax_normalized, y)
            
            if error < best_error:
                best_error = error
                best_x = x
                best_lambda = lambda_scale
    
    if best_x is None:
        raise ValueError("Could not find non-zero solution with Lasso")
        
    return best_x, best_error, best_lambda

def visualize(nonzero_indices):
    print("Only drawing the first 8 frames")
    
    _, axes = plt.subplots(2, 4, figsize=(20, 10))
    t = 0
    for i in range(2):
        for j in range(4):
            img_path = f"/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/images/_xMr-HKMfVA/frame_{nonzero_indices[t]:04d}.jpg"
            axes[i,j].imshow(Image.open(img_path))
            axes[i,j].axis('off')
            t += 1
    plt.savefig('/scratch3/kat049/all-seeing/all-seeing-v2/temp_sgc/lasso.jpg')

def user_annotations():
    import pandas as pd
    filepath = '/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'
    df = pd.read_csv(filepath, sep='\t', header=None, names=['video_id', 'category', 'frame_values'])
    video_data = {}
    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].copy() # Create a copy to avoid SettingWithCopyWarning
        
        # Convert frame_values to a list of lists of integers. Handle potential errors
        try:
            video_df['frame_values'] = video_df['frame_values'].apply(lambda x: [int(val) for val in x.split(',')])
        except ValueError:
            print(f"Error: Invalid frame values for video {video_id}. Skipping this video.")
            continue  # Skip to the next video if there's an error

        # Find the maximum number of frames to ensure consistent DataFrame shape
        max_frames = video_df['frame_values'].apply(len).max()
        
        frame_data = []
        for _, row in video_df.iterrows():
            frame_values = row['frame_values']
            # Pad with zeros if a row has fewer frame values than the maximum
            padded_values = frame_values + [0] * (max_frames - len(frame_values))  # Pad with 0s
            frame_data.append(padded_values)

        frame_array = np.array(frame_data)  # Convert to NumPy array for efficient calculations

        frame_sum = np.sum(frame_array, axis=0)
        frame_mean = np.mean(frame_array, axis=0)

        results_df = pd.DataFrame({'frame_sum': frame_sum, 'frame_mean': frame_mean})
        video_data[video_id] = results_df

    return video_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # eval_model(args)
    generate_pdf()
    # user_annotations()