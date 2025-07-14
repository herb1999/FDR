
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
import random
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class KeywordGrounding(nn.Module):
    def __init__(self, args, embed_dim, clip):
        super().__init__()
        # Get the parameters of the last layer of CLIP's text encoder
        last_layer = clip.transformer.resblocks[-1]

        # Reuse QKV projection matrices (CLIP uses merged in_proj)
        in_proj_weight = last_layer.attn.in_proj_weight  # shape: [3*dim, dim]
        q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

        # Initialize trainable parameters
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Assign parameter values
        self.query.weight.data = q_weight.detach().clone().float()
        self.key.weight.data = k_weight.detach().clone().float()
        self.value.weight.data = v_weight.detach().clone().float()

        # Freeze parameters
        if args.freeze_keyword_attention:
            self.query.weight.requires_grad_(False)
            self.key.weight.requires_grad_(False)
            self.value.weight.requires_grad_(False)

        self.scale = embed_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, all_token_features, eos_feature):
        """
        Explanation of new parameters:
            all_token_features: Filtered valid token features [num_valid_tokens, dim]
        """
        q = self.query(eos_feature.unsqueeze(0))  # (1, dim)
        k = self.key(all_token_features)  # (num_valid_tokens, dim)
        v = self.value(all_token_features)  # (num_valid_tokens, dim)

        # Calculate attention weights
        attn_scores = torch.matmul(q, k.T) / self.scale  # (1, num_valid_tokens)
        attn_weights = self.softmax(attn_scores)  # (1, num_valid_tokens)

        # Apply attention weighting on value
        v_scaled = v * attn_weights.T  # (num_valid_tokens, dim)

        # Apply softmax across each dimension
        v_scaled_softmax = F.softmax(v_scaled, dim=0)  # Normalize across the token dimension

        return v_scaled_softmax  # (num_valid_tokens, dim)

class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load CLIP model
        clip_model, preprocess = clip.load("ViT-B/16", device=self.device)
        self.clip_model = clip_model
        self.encoder = clip_model.visual
        self.num_features = 512  # Output feature dimension for ViT-B/16
        self.preprocess = preprocess
        # CLIP's BPE tokenizer
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()

        # Prototype
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        self.keyword_attention = KeywordGrounding(args, embed_dim=self.num_features, clip=clip_model).to(self.device)

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if self.args.soft_mode == 'keyword_seg_with_training':
            self.shift_weight = args.shift_weight
            self.softmax_t = args.softmax_t
        else:
            self.shift_weight = args.shift_weight
            self.softmax_t = args.softmax_t

        self.main_knowledge_base = {}  # Main knowledge base (final merge)
        self.stemmer = PorterStemmer()  # Need to import from nltk.stem

    def normalize_keyword(self, keyword):
        """Normalize keyword"""
        keyword = keyword.lower().replace("-", " ").replace("_", " ")
        words = keyword.split()
        stemmed = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed)

    def build_knowledge_base(self, args):
        """Build the base class keyword knowledge base"""
        class_info = self.load_csv(args.class_label)
        base_class_ids = list(range(args.base_class))
        keywords_total = 0

        for class_id in base_class_ids:
            proto = self.fc.weight.data[class_id]
            class_text = f"{class_info[class_id]['name']}: {class_info[class_id]['desc']}"
            keywords = class_info[class_id]['keywords']

            # Break down the prototype features
            keyword_indexes, matched_keywords = self.find_keyword_token_indexes(class_text, keywords)
            keywords_total += len(matched_keywords)
            if not keyword_indexes:
                continue

            # Analyze keyword impact (naive or detailed)
            if args.naive_kg:
                contributions = self.analyze_keyword_impact_naive(class_text, keyword_indexes)
            else:
                contributions = self.analyze_keyword_impact(class_text, keyword_indexes)

            for i, keyword in enumerate(matched_keywords):
                contrib = contributions[i]
                # Generate a mask for top-k dimensions
                if args.keyword_threshold == 0:
                    topk_dims = torch.topk(contrib.abs(), args.topk_keyword_dim).indices
                    mask = torch.zeros_like(proto, dtype=torch.bool)
                    mask[topk_dims] = True
                else:
                    # Generate mask using threshold filter
                    mask = (contrib.abs() > args.keyword_threshold)  # Filter with threshold

                # Normalize keyword
                norm_key = self.normalize_keyword(keyword)

                # Merge into knowledge base
                if norm_key in self.main_knowledge_base:
                    self.main_knowledge_base[norm_key] |= mask
                else:
                    self.main_knowledge_base[norm_key] = mask

        print(self.main_knowledge_base.keys())
        print(f'total:{keywords_total}, keys:{len(self.main_knowledge_base.keys())}')

    def encode(self, x):
        x = x.half()  # Convert input to FP16 to match CLIP parameters
        with torch.no_grad():  # Ensure no gradient computation during encoding
            x = self.encoder(x)  # CLIP ViT directly outputs (batch_size, 512)
        return x.float()  # Convert back to single precision

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.normalize(x, p=2, dim=-1)
            fc_weight = F.normalize(self.fc.weight, p=2, dim=-1)
            x = F.linear(x, fc_weight)
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session):
        # Incrementally update the prototypes in the current session
        for batch in dataloader:  # Batch size is the total number of new class samples, only one round
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def load_csv(self, csv_path):
        """Load CSV file and parse class information and keywords"""
        df = pd.read_csv(csv_path, delimiter=";")
        class_info = {}
        for _, row in df.iterrows():
            class_id = row["idx"]
            class_name = row["class_name"]
            description = row["description"]
            keywords = row["keywords"].split(", ")  # Comma-separated keywords
            class_info[class_id] = {"name": class_name, "desc": description,
                                    "keywords": keywords}
        return class_info

    def get_text_features(self, text):
        """Encode text features using CLIP"""
        tokens = clip.tokenize([text]).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        return text_features

    def find_keyword_token_indexes(self, text, keywords):
        tokenized_text = self.tokenizer.encode(text)
        sot_token = self.tokenizer.encode("<|startoftext|>")[0]

        keyword_index_groups = []
        matched_keywords = []

        for keyword in keywords:
            # Get the token sequence of the keyword (excluding special tokens)
            keyword_tokens = [t for t in self.tokenizer.encode(keyword)
                              if t not in [sot_token, 0, 49407]]  # Filter out SOT/PAD/EOS tokens

            matches = []
            # Sliding window matching (allowing overlap)
            for i in range(len(tokenized_text) - len(keyword_tokens) + 1):
                window = tokenized_text[i:i + len(keyword_tokens)]
                if window == keyword_tokens:
                    # Record all matching positions (consider the SOT offset in CLIP)
                    matches = [i]
                    break

            if matches:
                keyword_index_groups.append(matches)  # Remove duplicates
                matched_keywords.append(keyword)

        return keyword_index_groups, matched_keywords

    def analyze_keyword_impact(self, text, keyword_index_groups):
        """Calculate the impact of each keyword token on the `[EOS]` feature"""
        tokens = clip.tokenize([text]).to(self.device)
        text_embedding = self.clip_model.token_embedding(tokens).float()
        positional_embedding = self.clip_model.positional_embedding.float()
        x = text_embedding + positional_embedding
        x = x.permute(1, 0, 2).half()
        for layer in self.clip_model.transformer.resblocks[:-1]:
            x = layer(x)

        x = x.float()
        eos_index = (tokens[0] == 49407).nonzero(as_tuple=True)[0].item()
        eos_feature = x[eos_index, :, :].squeeze(0)  # (dim,)

        # Get features of all tokens and compress the dimension (excluding padding)
        all_token_features = x.squeeze(1)[1:eos_index]  # (seq_len, dim)

        # Get the contribution distribution of all tokens [seq_len, dim]
        all_contributions = self.keyword_attention(
            all_token_features,  # Only pass valid tokens
            eos_feature
        )

        # Initialize contribution tensor [num_keywords, dim]
        contributions = torch.zeros(
            len(keyword_index_groups),
            all_contributions.size(1),
            device=self.device
        )

        # Aggregate for each keyword group
        for i, orig_indices in enumerate(keyword_index_groups):
            contributions[i] = torch.sum(
                all_contributions[orig_indices],
                dim=0
            )

        return contributions

    def analyze_keyword_impact_naive(self, text, keyword_index_groups):
        tokens = clip.tokenize([text]).to(self.device)
        text_embedding = self.clip_model.token_embedding(tokens).float()

        eos_index = (tokens[0] == 49407).nonzero(as_tuple=True)[0].item()
        eos_feature = text_embedding[0, eos_index]  # (dim,)

        flattened_indices = [idx for group in keyword_index_groups for idx in group]
        all_token_features = text_embedding[0, 1:eos_index]

        keyword_features = all_token_features[flattened_indices]
        return keyword_features * eos_feature  # (num_all_tokens, dim)

    def keyword_seg_calibration(self, args, session):
        """Keyword-guided prototype calibration"""
        class_info = self.load_csv(args.class_label)  # Get class text information

        current_kb = self.main_knowledge_base

        base_class_ids = list(range(args.base_class))
        cur_class_ids = list(range(args.base_class + (session - 1) * args.way,
                                   args.base_class + session * args.way))

        # 1. **Get text features**
        text_features = {}
        for class_id in base_class_ids + cur_class_ids:
            class_text = f"{class_info[class_id]['name']}: {class_info[class_id]['desc']}"
            text_features[class_id] = self.get_text_features(class_text)

        # 2. Get **base class text features**
        base_protos_text = torch.stack([text_features[cid] for cid in base_class_ids]).to(self.device)
        base_protos_text = F.normalize(base_protos_text, p=2, dim=-1)

        # 3. Get **new class text features for the current session**
        cur_protos_text = torch.stack([text_features[cid] for cid in cur_class_ids]).to(self.device)
        cur_protos_text = F.normalize(cur_protos_text, p=2, dim=-1)

        # 4. **Calculate similarity weights of overall text features**
        weights = torch.mm(cur_protos_text, base_protos_text.T) * args.softmax_t
        norm_weights = torch.softmax(weights, dim=1).float()

        # 5. Get **visual prototypes of the base classes**
        base_protos = self.fc.weight.data[base_class_ids].detach()
        base_protos = F.normalize(base_protos, p=2, dim=-1)

        cur_protos = self.fc.weight.data[cur_class_ids].detach()
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)

        # ==================Coarse-grained calibration========================
        delta_protos = torch.matmul(norm_weights, base_protos)  # Compute the weighted sum
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)

        updated_protos = (1 - self.shift_weight) * cur_protos + self.shift_weight * delta_protos

        # ==================Fine-grained calibration========================
        for class_id in cur_class_ids:
            # 6.1 Get class text information
            class_text = f"{class_info[class_id]['name']}: {class_info[class_id]['desc']}"
            keywords = class_info[class_id]['keywords']
            relative_id = class_id - args.base_class - (session - 1) * args.way

            updated_proto = updated_protos[relative_id]

            for i, keyword in enumerate(keywords):
                # Get keyword dimensions using method one: knowledge base retrieval
                norm_key = self.normalize_keyword(keyword)
                if norm_key not in current_kb:
                    continue

                mask = current_kb[norm_key]
                top_k_dims = mask.nonzero().squeeze()

                # Get keyword dimensions using method two: self-calculation
                # top_k_dims = torch.topk(contributions[i].abs(), k=args.topk_keyword_dim).indices  # (top_k,)

                # **6.5 Extract text features for dimensions corresponding to the keyword**
                base_sub_text = base_protos_text[:, top_k_dims]  # (base_class, top_k)
                cur_sub_text = cur_protos_text[relative_id, top_k_dims]  # (top_k,)

                # **Compute the similarity of the keyword in these dimensions**
                sub_similarity = torch.cosine_similarity(cur_sub_text.unsqueeze(0), base_sub_text,
                                                         dim=-1)  # (base_class,)

                # **Only retain the top_k similarities, others set to 0**
                topk_values, topk_indices = torch.topk(sub_similarity, k=args.topk_similarity,
                                                       dim=0)  # Select top_k related classes
                filtered_similarity = torch.zeros_like(sub_similarity).to(self.device)  # Create a zero vector
                filtered_similarity[topk_indices] = topk_values  # Keep only the top_k class similarities

                # **Compute the weight of the keyword's influence**
                sub_weights = torch.softmax(filtered_similarity * args.softmax_t,
                                            dim=0).float()  # Normalize only for top_k

                # **6.6 Compute the calibration amount for the keyword's influence**
                delta_sub_protos = torch.matmul(sub_weights, base_protos[:, top_k_dims])  # (top_k,)

                updated_proto[top_k_dims] = (1 - self.shift_weight) * updated_proto[
                    top_k_dims] + self.shift_weight * delta_sub_protos
            updated_protos[relative_id] = updated_proto

        # 7. **Update the category prototypes in the model**
        self.fc.weight.data[cur_class_ids] = updated_protos
