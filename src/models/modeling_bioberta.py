import copy
from typing import Optional, Dict

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.utils import ModelOutput

from .modeling_mixedtab import MixedTabModel
from .modeling_ucad import UCADModel
from .lm_head import MixedTabLMHead

# Hawkes temporal attention — lazy import to avoid hard dependency.
# When use_hawkes_attention=True in config, HawkesUCADModel replaces UCADModel.
try:
    from allofus.hawkes_attention import (
        HawkesUCADModel,
        date_ids_to_days,
        assign_bos_timestamp,
    )
    _HAWKES_AVAILABLE = True
except ImportError:
    _HAWKES_AVAILABLE = False


class BioBERTa(RobertaPreTrainedModel):
    def __init__(self, config, encoder_num_hidden_layers=2):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.num_hidden_layers = encoder_num_hidden_layers

        self.lab_encoder = MixedTabModel(encoder_config)
        self.personal_encoder = MixedTabModel(encoder_config)
        self.drug_encoder = MixedTabModel(encoder_config)

    def forward(
        self,
        # INPUT IDs
        drug: Dict[str, torch.Tensor],
        lab: Dict[str, torch.Tensor],
        personal: Dict[str, torch.Tensor],
        # OTHER ARGS
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        drug_outputs = self.drug_encoder(
            **drug,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        lab_outputs = self.lab_encoder(
            **lab,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        personal_outputs = self.personal_encoder(
            **personal,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return {
            "drug": drug_outputs,
            "lab": lab_outputs,
            "personal": personal_outputs,
        }


class BioBERTaForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config, encoder_num_hidden_layers=2):
        super().__init__(config)
        self.encoder = BioBERTa(
            config, encoder_num_hidden_layers=encoder_num_hidden_layers
        )

        self.use_hawkes = getattr(config, "use_hawkes_attention", False)
        if self.use_hawkes:
            assert _HAWKES_AVAILABLE, (
                "use_hawkes_attention=True but allofus.hawkes_attention not importable. "
                "Ensure the mdd-analysis-pipeline/allofus dir is on sys.path."
            )
            self.fused_decoder = HawkesUCADModel(config)
        else:
            self.fused_decoder = UCADModel(config)
        self.use_encoder = config.use_encoder

        self.lab_lm_head = MixedTabLMHead(config)
        self.personal_lm_head = MixedTabLMHead(config)
        self.drug_lm_head = MixedTabLMHead(config)
        self.disease_lm_head = MixedTabLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        def tie_lm_head_weights(lm_head, embeddings):
            lm_head.decoder.weight = embeddings.word_embeddings.weight
            lm_head.float_predictor.weight = torch.nn.Parameter(embeddings.measurement_embeddings.weight.T)

        tie_lm_head_weights(
            self.lab_lm_head,
            self.encoder.lab_encoder.embeddings
        )

        tie_lm_head_weights(
            self.personal_lm_head,
            self.encoder.personal_encoder.embeddings
        )

        tie_lm_head_weights(
            self.drug_lm_head,
            self.encoder.drug_encoder.embeddings
        )

        tie_lm_head_weights(
            self.disease_lm_head,
            self.fused_decoder.embeddings
        )

    def calculate_loss(self, lm_head, sequence_output, measurement_mask, labels, outputs):
        measurement_mask = measurement_mask.bool()

        prediction_scores, float_preds = lm_head(sequence_output)

        float_labels = labels.clone()
        float_mask = measurement_mask & (labels != -100)

        token_labels = labels.clone()
        token_labels[measurement_mask] = -100
        token_labels = token_labels.int()

        # CCE loss for tokens
        token_labels = token_labels.to(prediction_scores.device)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            token_labels.long().view(-1),
        )

        # MSE loss for floats
        if measurement_mask.any():
            float_labels = float_labels.to(float_preds.device)
            float_loss_fct = MSELoss(reduction="none")
            float_loss = float_loss_fct(float_preds, float_labels) * float_mask
            float_loss = float_loss.sum() / float_mask.sum()
        else:
            float_loss = torch.tensor(0.0, device=float_preds.device)

        outputs["loss"] = masked_lm_loss + float_loss
        outputs["cce_loss"] = masked_lm_loss
        outputs["mse_loss"] = float_loss

        return outputs

    def forward(
        self,
        disease: Dict[str, torch.Tensor],
        drug: Dict[str, torch.Tensor],
        lab: Dict[str, torch.Tensor],
        personal: Dict[str, torch.Tensor],
        # OTHER ARGS
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_encoder_outputs: Optional[bool] = False,
    ):
        disease_labels = disease.pop("labels")
        drug_labels = drug.pop("labels")
        lab_labels = lab.pop("labels")
        personal_labels = personal.pop("labels")

        disease_float_loss_mask = disease.pop("float_loss_mask")
        drug_float_loss_mask = drug.pop("float_loss_mask")
        lab_float_loss_mask = lab.pop("float_loss_mask")
        personal_float_loss_mask = personal.pop("float_loss_mask")

        encoder_outputs: dict[str, ModelOutput] = self.encoder(
            drug=drug,
            lab=lab,
            personal=personal,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Compute raw-day timestamps for Hawkes temporal kernels
        hawkes_kwargs = {}
        if self.use_hawkes:
            disease_ts = date_ids_to_days(
                disease["year_ids"], disease["month_ids"], disease["day_ids"]
            )
            disease_ts = assign_bos_timestamp(
                disease_ts, disease["attention_mask"], method="cutoff"
            )
            drug_ts = date_ids_to_days(
                drug["year_ids"], drug["month_ids"], drug["day_ids"]
            )
            drug_ts = assign_bos_timestamp(
                drug_ts, drug["attention_mask"], method="cutoff"
            )
            lab_ts = date_ids_to_days(
                lab["year_ids"], lab["month_ids"], lab["day_ids"]
            )
            lab_ts = assign_bos_timestamp(
                lab_ts, lab["attention_mask"], method="cutoff"
            )
            hawkes_kwargs = dict(
                disease_timestamps=disease_ts,
                drug_timestamps=drug_ts,
                lab_timestamps=lab_ts,
            )

        fused_outputs = self.fused_decoder(
            # MAIN INPUTS
            **disease,
            # HIDDEN STATES
            drug_hidden_states=encoder_outputs["drug"][0],
            lab_hidden_states=encoder_outputs["lab"][0],
            personal_hidden_states=encoder_outputs["personal"][0],
            # ATTENTION MASKS
            drug_attention_mask=drug["attention_mask"],
            lab_attention_mask=lab["attention_mask"],
            personal_attention_mask=personal["attention_mask"],
            # TIMESTAMPS (Hawkes only)
            **hawkes_kwargs,
            # OTHER ARGS
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        fused_outputs = self.calculate_loss(
            self.disease_lm_head,
            fused_outputs[0],
            disease_float_loss_mask,
            disease_labels,
            fused_outputs,
        )

        encoder_outputs["drug"] = self.calculate_loss(
            self.drug_lm_head,
            encoder_outputs["drug"][0],
            drug_float_loss_mask,
            drug_labels,
            encoder_outputs["drug"],
        )

        encoder_outputs["lab"] = self.calculate_loss(
            self.lab_lm_head,
            encoder_outputs["lab"][0],
            lab_float_loss_mask,
            lab_labels,
            encoder_outputs["lab"],
        )

        encoder_outputs["personal"] = self.calculate_loss(
            self.personal_lm_head,
            encoder_outputs["personal"][0],
            personal_float_loss_mask,
            personal_labels,
            encoder_outputs["personal"],
        )

        fused_outputs["total_loss"] = fused_outputs["loss"] + encoder_outputs["drug"]["loss"] + encoder_outputs["lab"]["loss"] + encoder_outputs["personal"]["loss"]

        if return_encoder_outputs:
            return fused_outputs, encoder_outputs

        return fused_outputs
