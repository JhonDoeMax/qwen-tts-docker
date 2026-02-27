from typing import Optional
import torch


def _build_prefill(
    qwen_model,
    input_id: torch.Tensor,
    *,
    instruct_id: Optional[torch.Tensor],
    ref_id: Optional[torch.Tensor],
    voice_clone_prompt: Optional[dict],
    language: str,
    speaker: Optional[str],
    non_streaming_mode: bool,
):
    """Create Talker prefill embeddings/masks for a single sample, following upstream generate()."""
    talker = qwen_model.talker

    pieces = []

    # Speaker embedding (CustomVoice / VoiceClone)
    voice_clone_spk_embeds = None
    if voice_clone_prompt is not None:
        voice_clone_spk_embeds = qwen_model.generate_speaker_prompt(voice_clone_prompt)

    if instruct_id is not None:
        pieces.append(talker.text_projection(talker.get_text_embeddings()(instruct_id)))

    if voice_clone_spk_embeds is None:
        if speaker == "" or speaker is None:
            speaker_embed = None
        else:
            if speaker.lower() not in qwen_model.config.talker_config.spk_id:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            spk_id = qwen_model.config.talker_config.spk_id[speaker.lower()]
            speaker_embed = talker.get_input_embeddings()(
                torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
            )
    else:
        if bool(voice_clone_prompt["x_vector_only_mode"][0]) or bool(voice_clone_prompt["icl_mode"][0]):
            speaker_embed = voice_clone_spk_embeds[0]
        else:
            speaker_embed = None

    # Language token id / dialect handling
    language_id = None
    if language is None:
        raise ValueError("language must be provided")
    if language.lower() != "auto":
        if language.lower() not in qwen_model.config.talker_config.codec_language_id:
            raise NotImplementedError(f"Language {language} not implemented")
        language_id = qwen_model.config.talker_config.codec_language_id[language.lower()]

    if (
        language.lower() in ["chinese", "auto"]
        and speaker not in ("", None)
        and qwen_model.config.talker_config.spk_is_dialect[speaker.lower()] is not False
    ):
        dialect = qwen_model.config.talker_config.spk_is_dialect[speaker.lower()]
        language_id = qwen_model.config.talker_config.codec_language_id[dialect]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[qwen_model.config.tts_bos_token_id, qwen_model.config.tts_eos_token_id, qwen_model.config.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            qwen_model.config.talker_config.codec_nothink_id,
            qwen_model.config.talker_config.codec_think_bos_id,
            qwen_model.config.talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            qwen_model.config.talker_config.codec_think_id,
            qwen_model.config.talker_config.codec_think_bos_id,
            language_id,
            qwen_model.config.talker_config.codec_think_eos_id,
        ]]

    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[qwen_model.config.talker_config.codec_pad_id, qwen_model.config.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )

    if speaker_embed is None:
        codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
    else:
        codec_input_embedding = torch.cat([codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1)

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    prefill_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]

    talker_input_embed = torch.cat((role_embed, prefill_embed), dim=1)

    # ICL prompt for Base (voice clone)
    if (
        voice_clone_prompt is not None
        and voice_clone_prompt.get("ref_code", None) is not None
        and voice_clone_prompt["ref_code"][0] is not None
        and bool(voice_clone_prompt["icl_mode"][0])
    ):
        if ref_id is None:
            raise ValueError("ref_text/ref_id is required for Base task in ICL mode (x_vector_only_mode=False)")
        icl_input_embed, trailing_text_hidden = qwen_model.generate_icl_prompt(
            text_id=input_id[:, 3:-5],
            ref_id=ref_id[:, 3:-2],
            ref_code=voice_clone_prompt["ref_code"][0].to(talker.device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
    else:
        # First text token
        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:],
            ],
            dim=1,
        )

        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            text_embed = torch.cat(
                (talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed),
                dim=1,
            )
            text_embed = text_embed + talker.get_input_embeddings()(
                torch.tensor(
                    [[qwen_model.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                    device=talker.device,
                    dtype=input_id.dtype,
                )
            )
            bos_embed = tts_pad_embed + talker.get_input_embeddings()(
                torch.tensor([[qwen_model.config.talker_config.codec_bos_id]], device=talker.device, dtype=input_id.dtype)
            )
            talker_input_embed = torch.cat([talker_input_embed, text_embed, bos_embed], dim=1)
            trailing_text_hidden = tts_pad_embed
        else:
            trailing_text_hidden = torch.cat(
                (talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed),
                dim=1,
            )

    pieces.append(talker_input_embed)
    talker_input_embeds = torch.cat(pieces, dim=1)
    attention_mask = torch.ones((1, talker_input_embeds.shape[1]), device=talker_input_embeds.device, dtype=torch.long)

    return talker_input_embeds, attention_mask, trailing_text_hidden, tts_pad_embed