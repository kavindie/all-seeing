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

    # x_optimal = binary_linear_PULP(frame_embeddings_norm.to('cpu').T, video_embedding_norm.to('cpu'))
    # X = solve_sparse_binary_greedy(frame_embeddings_norm.to('cpu').T, video_embedding_norm.to('cpu'))
    x = solve_with_lasso(frame_embeddings_norm.to('cpu').T, video_embedding_norm.to('cpu'))


    downsample_num = 240
    frame_embeddings_norm_subset = frame_embeddings_norm[::downsample_num]

    selected_combo =  l0_optimization(frame_embeddings_norm_subset.to('cpu').T, video_embedding_norm.to('cpu'))
    selected_indices_in_original = (np.array(selected_combo).nonzero()[0] * downsample_num).tolist()
    l0_simple_img_num = selected_indices_in_original[0] - 1 # need to iterate for other combos
    l0_simple_img_path = f"/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/images/_xMr-HKMfVA/frame_{l0_simple_img_num:04d}.jpg"
    Image.open(l0_simple_img_path).save(f'/scratch3/kat049/all-seeing/all-seeing-v2/temp_sgc/l0_img.jpg')

    # array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def solve_sparse_binary_greedy(A, y, tolerance=1e-6):
    """
    Solve Ax = y where x is binary using a greedy approach
    Finds minimal set of vectors from A that best approximate y
    
    Parameters:
    A: numpy array of shape [m, n] - frame embeddings
    y: numpy array of shape [m] - video embedding
    tolerance: error tolerance for solution
    
    Returns:
    x: binary solution vector
    error: final error
    selected_indices: indices of selected frames
    """
    A = A.type(torch.float32)
    y = y.type(torch.float32)
    m, n = A.shape
    
    max_ones = None
    best_x = None
    best_error = float('inf')
    best_lambda = None
    
    # If max_ones is not specified, try up to n/2 ones
    if max_ones is None:
        max_ones = n // 2
    
    # Try combinations with increasing number of ones
    for num_ones in range(1, max_ones + 1):
        # Try all possible combinations of num_ones positions
        for ones_positions in combinations(range(n), num_ones):
            # Create binary vector
            x = np.zeros(n)
            x[list(ones_positions)] = 1
            
            # Calculate Ax
            Ax = A @ x
            
            # Calculate scaling factor lambda to make λAx unit length
            lambda_scale = 1.0 / np.linalg.norm(Ax)
            
            # Scale Ax to unit length
            Ax_normalized = Ax * lambda_scale
            
            # Calculate error (1 - cosine similarity)
            error = 1.0 - np.dot(Ax_normalized, y)
            
            # Update best solution if this is better
            if error < best_error:
                best_error = error
                best_x = x
                best_lambda = lambda_scale
                
                # If error is very small, we can stop
                if error < 1e-10:
                    return best_x, best_error, best_lambda

    # m, n = A.shape
    # x = torch.zeros(n)
    # residual = y.clone()
    # residual = residual.type(torch.float32)
    # selected_indices = []

    # while True:
    #     # Compute correlation between residual and remaining vectors
    #     scores = torch.abs(A.T @ residual)
        
    #     # Zero out already selected indices
    #     scores[selected_indices] = 0
        
    #     # Find best matching vector
    #     best_idx = torch.argmax(scores).item()
        
    #     # If we're not improving significantly, stop
    #     if scores[best_idx] < tolerance:
    #         break
            
    #     # Add the vector
    #     selected_indices.append(best_idx)
    #     x[best_idx] = 1
        
    #     # Update residual
    #     current_approx = A @ x
    #     residual = y - current_approx
        
    #     # Check if we're close enough
    #     error = np.linalg.norm(residual)
    #     if error < tolerance:
    #         break
            
    # return x, np.linalg.norm(A @ x - y), selected_indices

def solve_sparse_binary(A, y, max_iter=1000):
    """
    Solve Ax = y where x is binary and we want minimum number of 1s
    
    Parameters:
    A: numpy array of shape [m, n]
    y: numpy array of shape [m]
    max_iter: maximum number of iterations
    
    Returns:
    x: binary solution vector
    error: final error
    """
    m, n = A.shape
    
    # Initialize CVXPY problem
    x = cp.Variable(n, boolean=True)
    objective = cp.sum(x)  # Minimize number of 1s
    
    # Constraints
    constraints = [
        cp.norm(A @ x - y) <= 10  # Ax ≈ y
    ]
    
    # Define and solve the problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        # Try solving with MOSEK if available (handles binary variables well)
        prob.solve(solver=cp.MOSEK)
    except:
        # Fall back to default solver
        prob.solve()
    
    if prob.status != cp.OPTIMAL:
        # If no optimal solution found, try relaxed version
        return solve_relaxed(A, y, max_iter)
    
    return x.value, prob.value

def solve_relaxed(A, y, max_iter):
    """
    Solve relaxed version if binary optimization fails
    Uses continuous optimization and then thresholds
    """
    m, n = A.shape
    
    # Initialize with continuous values
    x = cp.Variable(n)
    objective = cp.norm1(x)  # L1 norm promotes sparsity
    
    constraints = [
        cp.norm(A @ x - y) <= 1e-6,
        0 <= x, x <= 1  # Box constraints
    ]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(max_iter=max_iter)
    
    # Threshold to get binary solution
    x_binary = np.where(x.value > 0.5, 1, 0)
    
    # Compute final error
    error = np.linalg.norm(A @ x_binary - y)
    
    return x_binary, error

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

def binary_linear_PULP(A, y):
    """
    A: [m, n] = frame_embeddings_norm
    y: [m] = video_embedding_norm
    """
    m, n = A.shape[0], A.shape[1]
    # Create the problem
    prob = pulp.LpProblem("Minimum_Vectors", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("Vector", range(n), cat='Binary')

    # Objective function
    prob += pulp.lpSum([x[i] for i in range(n)])

    # Constraints
    # for i in range(m):
    #     prob += pulp.lpSum([A[i, j].item() * x[j] for j in range(n)]) == y[i].item() 
    for i in tqdm(range(m), desc="Adding constraints"):  # Add tqdm here
        prob += pulp.lpSum([A[i, j].item() * x[j] for j in range(n)]) == y[i].item() 
    # Solve the problem
    prob.solve()

    # Print the status of the solution
    print("Status:", pulp.LpStatus[prob.status])

    # Print the optimal solution
    if prob.status == pulp.LpStatusOptimal:
        x_optimal = np.array([x[i].varValue for i in range(n)])
        print("Optimal x:", x_optimal)
        print("Number of vectors used:", np.sum(x_optimal))
        return x_optimal
    else:
        print("Optimization failed.")
        return None

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
