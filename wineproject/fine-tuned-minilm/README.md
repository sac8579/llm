---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:39988
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Decanting recommended for a balanced, buttery and full-bodied white
  sentences:
  - Dusty cherry and tomato aromas are neither elegant nor elevated. A jagged palate
    lacks flow, while this tastes of salty sour plum. A low-depth finish tastes peppery.
    Overall, this is simply adequate Pinot Noir.
  - Pale in color, this fragrant wine has some bottle age that gives a toast and almond
    character. Ripe strawberry flavors balance between softness and a crisp, tight
    aftertaste. It is ready to drink.
  - Starts off slowly, but opens considerably with air—so much so that it's probably
    worth decanting as you might a bottle of young white Burgundy. It's buttery and
    full, yet possesses a stony mineral quality that gives it balance. Initial peppery-spicy
    notes on the finish smooth out the longer it sits in the glass. Editors' Choice.
- source_sentence: Deep red wine, raspberry and vanilla scent
  sentences:
  - Features some superb elements that come off as a bit disjointed—perhaps it just
    needs some time? High-toned cherry fruit picks up some herbal notes and peppery
    cinnamon flavors, then finishes crisp, with hints of orange pekoe tea.
  - This wine has a deep violet-red color and a nose of raspberry, vanilla and graphite.
    Fruit-dominant flavors of raspberry and cherry, with a touch of custard, showcase
    a satisfying cooperation between deep and bright notes that make for a satisfying
    sipping experience.
  - Wiese & Krohn have made a specialty of Colheitas, vintage-dated wood-aged Ports.
    This 1983, aged in wood for 28 years, is an opulent, ripe example of the style.
    It is sweet and very concentrated, with burnt old wood flavors to go with raisins
    and prunes. The acidity is intense, a fine counterpoint to the rich fruit.
- source_sentence: somthing that is Light, citrus and peach chardonnay
  sentences:
  - This is a rich, full-bodied and finely structured wine. Named after the patriarch
    of the estate, it is firmly packed with dark plum and ripe blackberry fruits that
    are flanked by a dense texture and balanced acidity. It needs some time, so drink
    from 2018.
  - Light in weight and subtle in flavor, this Chard is a welcome change from some
    of its mega-mallowy Oz compatriots. There are citrus and peach flavors on the
    palate and floral, yellow fruit aromas; it starts to get toasty with aeration.
  - This blend of 45% Cabernet Sauvignon, 45% Merlot and 10% Syrah was too sweet for
    this taster. It's full of huge ripe fruit and a lingering aroma of burnt toast.
- source_sentence: Wine with Complex wine, surprising aroma and transitions to black
    fruits
  sentences:
  - This pure Merlot is racy, ripe and ready to drink, with fruit that shows maturity.
    Tannins are well integrated into the structure, leaving plenty of room for the
    juicy fruit and mature, spicy flavors. Drink now.
  - Forced, candied aromas of chocolaty oak and herbal, slightly weedy berry fruits
    feed into a chunky, blocky, creamy palate with saucy berry flavors combined with
    minty, top-heavy oak. The finish tastes minty, lactic and oaky, with heat.
  - This wine opens with a curious aroma of fried foods, like hashed browns from a
    diner, before finding its sense of black fruit, mild spice and leather. Dry and
    somewhat puckering on the palate, it's full bodied and powerfully concentrated.
- source_sentence: somthing that is Delicate red, fruity notes and a touch of oakiness
  sentences:
  - Sourced from the winery's highest, steepest site, and barrel-fermented with partial
    malolactic, this is a fine effort in a lighter vintage. Yeasty and creamy in the
    mouth, it melds a mix of banana, pineapple and coconut cream in a wine with excellent
    structure and length. The hint of caramel provides a flourish on the finish.
  - This tiny boutique does well with thick and heavy wines that frequently top 15%
    alcohol. This sweet, syrupy Syrah is loaded with blueberries, chocolate, licorice
    and espresso flavors. Nothing shy here, but it delivers a lot of flavor for the
    price.
  - This is a dark, earthy wine. Clove and leather notes accent modest black-cherry
    fruit, but the texture is plush and silky, making it approachable now.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'somthing that is Delicate red, fruity notes and a touch of oakiness',
    'This is a dark, earthy wine. Clove and leather notes accent modest black-cherry fruit, but the texture is plush and silky, making it approachable now.',
    'This tiny boutique does well with thick and heavy wines that frequently top 15% alcohol. This sweet, syrupy Syrah is loaded with blueberries, chocolate, licorice and espresso flavors. Nothing shy here, but it delivers a lot of flavor for the price.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 39,988 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | float                                                          |
  | details | <ul><li>min: 8 tokens</li><li>mean: 14.68 tokens</li><li>max: 54 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 58.28 tokens</li><li>max: 127 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.51</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Looking for a Fruity, juicy red wine and a crisp note wine</code> | <code>Though reserved on the nose at first, this bottling begins to show dense blueberry, coffee and soy with patience. There are boisterous flavors of squeezed elderberry and blackberry fruit once sipped, along with beef char, black coffee, dark chocolate and purple flowers on the finish. This still needs a little time to open, but shows lots of potential.</code> | <code>0.0</code> |
  | <code>A Complex, mineral-rich white and apple wine</code>               | <code>Absorbingly complex with racy, textural, mineral-driven flavors, this excellent wine features fleshy apple and pear fruit, crisply defined and extended. The low alcohol—just a tad over 13%—keeps the acids front and center; the wine is immaculately fresh, with just a touch (0.7%) of residual sugar.</code>                                                        | <code>1.0</code> |
  | <code>Crisp, fruity white and medium body</code>                        | <code>This smooth, wood-aged wine is ripe with red fruits, soft tannins and a generous, full-bodied character. The acidity keeps the wine in shape, round and ready to drink.</code>                                                                                                                                                                                           | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step  | Training Loss |
|:-----:|:-----:|:-------------:|
| 0.2   | 500   | 0.1157        |
| 0.4   | 1000  | 0.0835        |
| 0.6   | 1500  | 0.0775        |
| 0.8   | 2000  | 0.0726        |
| 1.0   | 2500  | 0.0712        |
| 1.2   | 3000  | 0.063         |
| 1.4   | 3500  | 0.0616        |
| 1.6   | 4000  | 0.0601        |
| 1.8   | 4500  | 0.0601        |
| 2.0   | 5000  | 0.06          |
| 2.2   | 5500  | 0.0534        |
| 2.4   | 6000  | 0.0536        |
| 2.6   | 6500  | 0.0524        |
| 2.8   | 7000  | 0.0527        |
| 3.0   | 7500  | 0.053         |
| 3.2   | 8000  | 0.048         |
| 3.4   | 8500  | 0.0478        |
| 3.6   | 9000  | 0.0492        |
| 3.8   | 9500  | 0.0484        |
| 4.0   | 10000 | 0.0467        |
| 4.2   | 10500 | 0.0436        |
| 4.4   | 11000 | 0.0461        |
| 4.6   | 11500 | 0.0444        |
| 4.8   | 12000 | 0.044         |
| 5.0   | 12500 | 0.0446        |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 4.0.1
- Transformers: 4.50.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.4.1
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->