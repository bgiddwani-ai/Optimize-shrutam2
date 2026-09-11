
=============
== PyTorch ==
=============

NVIDIA Release 25.04 (build 159049541)
PyTorch Version 2.7.0a0+79aa174
Container image Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2014-2024 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

GOVERNING TERMS: The software and materials are governed by the NVIDIA Software License Agreement
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the Product-Specific Terms for NVIDIA AI Products
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...

                   input_text,
                   audios,
                   audio_ids,
                   args,
                   stream,
                   history=None,
                   past_audio_features=None,
                   run_time=1):
        assert input_text, "input_text must be provided"
        assert torch.cuda.is_available(), "no gpu available"
        # preprocess on CPU maybe faster
        device = torch.device("cpu")
        if isinstance(history, list):
            history.append(input_text)
            full_text = self.processor.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False)
        else:
            full_text = input_text
        inputs = self.processor(
            text=full_text,
            audios=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs = inputs.to(device)
        input_ids = inputs.input_ids

        if hasattr(inputs,
                   'input_features') and inputs.input_features is not None:
            # audio tower
            batch_size, _, max_mel_seq_len = inputs.input_features.shape
            feature_attention_mask = inputs.feature_attention_mask

            audio_feat_lengths, num_audio_tokens = self.get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))

            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (torch.arange(0,
                                      max_seq_len,
                                      dtype=audio_feat_lengths.dtype,
                                      device=device).unsqueeze(0).expand(
                                          batch_size, max_seq_len))
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len).expand(batch_size, 1,
                                                      max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(dtype=torch.float16,
                                                            device=device)
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_features = self.audio_tower(inputs.input_features,
                                              audio_attention_mask, stream,
                                              run_time)

            # merge audio features and input ids
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(
                max_audio_tokens, device=device).expand(
                    num_audios,
                    max_audio_tokens) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(
                -1, embed_dim)
            batch_size, _ = input_ids.shape

            # 1. Create a mask to know where special audio tokens are
            special_audio_token_mask = input_ids == self.config.audio_token_index
            special_audio_token_num = special_audio_token_mask.sum().item()
            if past_audio_features is not None:
                assert isinstance(past_audio_features,
                                  list), f'past_audio_features should be a list'
                assert (
                    special_audio_token_num == len(past_audio_features) +
                    num_audios
                ), f'special_audio_token_num {special_audio_token_num} should be equal to len(past_audio_features) + num_audios ({len(past_audio_features)} + {num_audios})'
                # split to get current audio features
                cur_audio_features = torch.split(masked_audio_features,
                                                 num_audio_tokens.tolist())
                if len(past_audio_features) > 0:
                    # concat past and current audio features
                    masked_audio_features = torch.cat(
                        (torch.cat(past_audio_features).to(
                            masked_audio_features.device),
                         masked_audio_features))
                    # get past audio tokens number
                    past_num_audio_tokens = torch.tensor([
                        past_feat.size(0) for past_feat in past_audio_features
                    ])
                    # concat past and current audio tokens number
                    num_audio_tokens = torch.cat(
                        (past_num_audio_tokens.to(num_audio_tokens.device),
                         num_audio_tokens))
                # extend past audio features, cache them in CPU memory
                past_audio_features.extend(
                    [cur_feat.cpu() for cur_feat in cur_audio_features])

            batch_indices, non_audio_indices = torch.where(
                input_ids != self.config.audio_token_index)

            # 2. Fill the final input ids based on the mask.
            batch_indices, audio_indices = torch.where(
                input_ids == self.config.audio_token_index)

            vocab_size = self.config.vocab_size
            fake_prompt_id = torch.arange(vocab_size,
                                          vocab_size + num_audio_tokens.sum(),
                                          device=device)

            input_ids[batch_indices, audio_indices] = fake_prompt_id
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.gpu_device)
            dtype = self.model_config.dtype
            prompt_table, tasks, task_vocab_size = self.ptuning_setup(
                masked_audio_features, dtype, embed_dim, None, input_ids)

            # build extra ids
            assert isinstance(audio_ids, list), "audio_ids must be a list"
            assert (
                len(audio_ids) == num_audio_tokens.size(0)
            ), f"audio_ids length doesn't match with num_audio_tokens ({len(audio_ids)} != {num_audio_tokens.size(0)})"
            for i in audio_ids:
                assert isinstance(
                    i, int
                ) and i > 0, "audio_id should be an integer greater than 0"
            extra_ids = torch.zeros_like(input_ids,
                                         dtype=torch.int64,
                                         device=device)
            seq_extra_ids = torch.cat([
                torch.full((n, ), audio_ids[i], dtype=torch.int64)
                for i, n in enumerate(num_audio_tokens)
            ]).to(device)
            extra_ids[batch_indices, audio_indices] = seq_extra_ids
            extra_ids = extra_ids.tolist()
        else:
            input_ids = input_ids.to(dtype=torch.int32, device=self.gpu_device)
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.gpu_device)
            dtype = self.model_config.dtype
            prompt_table, tasks, task_vocab_size = self.ptuning_setup(
                None, dtype, self.model_config.hidden_size, None, input_ids)
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64).tolist()

        # print(f"extra_ids: {extra_ids}")
        output_ids, Qwen_time = self.generate_for_qwen_audio(
            input_ids, args, prompt_table, extra_ids, run_time)

        runtime_rank = tensorrt_llm.mpi_rank()
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device=self.gpu_device,
                                     dtype=torch.int32)
        effective_output_token = 0
        if runtime_rank == 0:
            if self.output_csv is None and self.output_npy is None:
                for b in range(input_lengths.size(0)):
                    inputs = input_ids[b]
                    if self.num_beams <= 1:
                        outputs = output_ids[b][0, len(inputs):].tolist()
                        try:
                            effective_output_token = (effective_output_token +
                                                      outputs.index(151643))
                        except:
                            effective_output_token = 1
                        output_text = self.tokenizer.decode(
                            outputs, skip_special_tokens=True)
                        print(f'Output: "{output_text}"')
                    else:
                        for beam in range(self.num_beams):
                            outputs = output_ids[b][beam, len(inputs):].tolist()
                            output_text = self.tokenizer.decode(
                                outputs, skip_special_tokens=True)
                            print(f'Output(beam: {beam}): "{output_text}"')
        logger.info(f"Input length={input_lengths[b]}")
        logger.info(f"Output length={output_ids.shape}")
        logger.info(f"TensorRT-LLM QWen time: {Qwen_time:3f} sec ")
        if isinstance(history, list):
            history.append({'role': 'assistant', 'content': output_text})
        return output_text, past_audio_features


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument(
        "--audio_engine_path",
        type=str,
