from open_pref_eval.datasets import ds2name, load_dataset_n

def load_eval_ds(path, N=None, split='test', name=None):
    """
    Load a dataset with a few shorthands
    - wassname/genies_preferences:justice#test
    - math -> "wassname/genies_preferences:math#test"
    """
    if '/' in path:
        if ':' in path:
            path, name = path.split(':')
        else:
            name = None
        if '#' in path:
            path, split = path.split('#')
        ds = load_dataset_n(path=path, name=name, split=split, N=N)
    else:
        ds =  load_dataset_n(
            "wassname/genies_preferences", name=path, split=split, N=N
        )
    return ds
TRAINING_EXPERIMENTS = {
    # PRIORITY 1: Core experiments for main paper
    "math": [
        {"target": "math", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "change_my_view", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "math_fiction", "type": "cross_domain", "label": "context", "category": "extreme"},  # context change is really cross-domain
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "cooking", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:commonsense", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/ethics_expression_preferences:utilitarianism", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/ethics_expression_preferences:deontology", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],


    "code": [
        {"target": "code", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "us_history", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "change_my_view", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "counterfactual_python", "type": "cross_domain", "label": "pretraining_similarity", "category": "extreme"},  # fold into cross_domain
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "alpaca_mmlu": [
        {"target": "alpaca_mmlu", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "truthful_qa", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "unhelpful_alpaca", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "ranking_logic", "type": "cross_domain", "label": "pretraining_similarity", "category": "extreme"},  # fold into cross_domain
        {"target": "spanish_output", "type": "cross_domain", "label": "encoding", "category": "extreme"},  # fold into cross_domain
        {"target": "comma_separated_output", "type": "cross_domain", "label": "encoding", "category": "extreme"},
        {"target": "sycophancy_mimicry", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "sycophancy_answer", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "crt_1", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "reward_seeking", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "survival_influence", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "personality_traits", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "gender_bias", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "wrong_arc", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "punishment_avoidance", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "crt_2", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "crt_3", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "sycophancy_feedback", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "spanish_input", "type": "cross_domain", "label": "encoding", "category": "extreme"},
        {"target": "comma_separated_input", "type": "cross_domain", "label": "encoding", "category": "extreme"},
        {"target": "raven_matrices", "type": "cross_domain", "label": "pretraining_similarity", "category": "extreme"},
        {"target": "word_swap", "type": "cross_domain", "label": "pretraining_similarity", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/ethics_expression_preferences:commonsense", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],

    "math_easy": [
        {"target": "math_easy", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "math_hard", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "alpaca_low_quality": [
        {"target": "alpaca_low_quality", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "alpaca_high_quality", "type": "difficulty_scaling", "label": "quality", "category": "extreme"},  # quality is like difficulty
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],

    "wassname/ethics_expression_preferences:justice": [
        {"target": "wassname/ethics_expression_preferences:justice", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "math", "type": "cross_domain", "label": "moral_to_technical"},
        {"target": "code", "type": "cross_domain", "label": "moral_to_technical"}, 
        {"target": "wassname/ethics_expression_preferences:commonsense", "type": "in_domain"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control"},
    ],


    
    # PRIORITY 2: Supporting experiments  

    "shp_low_quality": [
        {"target": "shp_low_quality", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "math", "type": "cross_domain", "label": "quality_to_technical"},
        {"target": "alpaca_mmlu", "type": "cross_domain", "label": "quality_to_factual"},
        {"target": "shp_high_quality", "type": "in_domain"}, 
        {"target": "wassname/medical-dpo-V2#data", "type": "control"},
    ],

    "cooking": [
        {"target": "cooking", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "math", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "raven_matrices", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/ethics_expression_preferences:commonsense", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "code_easy": [
        {"target": "code_easy", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "code_hard", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "us_history": [
        {"target": "us_history", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "code", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "math", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "us_history_textbook", "type": "cross_domain", "label": "context", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "raven_matrices": [
        {"target": "raven_matrices", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "us_history", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "code", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "change_my_view": [
        {"target": "change_my_view", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "raven_matrices", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "cooking", "type": "cross_domain", "label": "skill", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    # PRIORITY 3: Comprehensive experiments (for appendix)
    "alpaca_easy": [
        {"target": "alpaca_easy", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "alpaca_hard", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "arc_easy": [
        {"target": "arc_easy", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "arc_hard", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "us_history_textbook": [
        {"target": "us_history_textbook", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "us_history_fiction", "type": "cross_domain", "label": "context", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    "alpaca_chat": [
        {"target": "alpaca_chat", "type": "in_domain", "label": "baseline", "category": "control"},
        {"target": "sycophancy_are_you_sure", "type": "alignment_robustness", "label": "unwanted_personas", "category": "probing"},
        {"target": "illegal_dont_help", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"},
    ],
    
    # Lower priority but comprehensive
    "raven_easy": [
        {"target": "raven_matrices", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "ranking_logic_easy": [
        {"target": "ranking_logic_hard", "type": "difficulty_scaling", "label": "difficulty", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "code_low_quality": [
        {"target": "code", "type": "difficulty_scaling", "label": "quality", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "shp_low_quality": [
        {"target": "shp_high_quality", "type": "difficulty_scaling", "label": "quality", "category": "extreme"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "alpaca_short": [
        {"target": "alpaca_long", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "pursue_goals": [
        {"target": "relinquish_power", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
    "creative_writing": [
        {"target": "biology_with_literary_style", "type": "alignment_robustness", "label": "spurious_cues", "category": "probing"},
        {"target": "wassname/ethics_expression_preferences:justice", "type": "moral_transfer", "label": "moral_alignment", "category": "alignment"},
        {"target": "wassname/medical-dpo-V2#data", "type": "control", "label": "control", "category": "control"}
    ],
}

# Clean table columns for paper:
TABLE_COLUMNS = {
    "in_domain": "In-Domain",           # baseline performance  
    "cross_domain": "Cross-Domain",     # core generalization claim
    "difficulty_scaling": "Difficulty", # scaling robustness
    "moral_transfer": "Moral Transfer", # alignment generalization  
    "alignment_robustness": "Safety",   # robustness to alignment failures
    "control": "Control"                # negative control
}
