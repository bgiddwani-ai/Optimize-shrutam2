
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

                sink_token_length=args.sink_token_length,
                max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=args.
                kv_cache_free_gpu_memory_fraction,
                cross_kv_cache_fraction=None,
                enable_chunked_context=args.enable_chunked_context,
                multi_block_mode=args.multi_block_mode,
                cuda_graph_mode=args.cuda_graph_mode,
                device_ids=[args.gpu_id])
        runner_kwargs.update(
            enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
        self.runner = runner_cls.from_dir(**runner_kwargs)
        self.tokenizer = tokenizer
        self.processor = processor
        self.sampling_config = sampling_config
        self.model_config = model_config

    def ptuning_setup(self, prompt_table, dtype, hidden_size, tasks, input_ids):
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[0]],
                                           dtype=torch.int32,
                                           device=self.gpu_device)
            prompt_table = prompt_table.to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device=self.gpu_device)
        else:
            prompt_table = torch.empty([1, hidden_size], device=self.gpu_device)
            task_vocab_size = torch.zeros([1], device=self.gpu_device)

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(",")],
                                 dtype=torch.int32,
                                 device=self.gpu_device)
            assert (tasks.shape[0] == input_ids.shape[0]
                    ), "Number of supplied tasks must match input batch size"
        else:
            tasks = torch.zeros([input_ids.size(0)],
                                dtype=torch.int32,
                                device=self.gpu_device)

        return [prompt_table, tasks, task_vocab_size]

    def build_user_input(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(
            text, str), "audio or text must be provided as user input"
        content = []
        if audio:
            content.append({'type': 'audio', 'audio_url': audio})
        if text:
            content.append({'type': 'text', 'text': text})
        user_input = {'role': 'user', 'content': content}
        return user_input

    def get_raw_audios(self, audio_url):
        audios = []
        for url in audio_url:
            if os.path.isfile(url):
                audio_data, _ = librosa.load(
                    url, sr=self.processor.feature_extractor.sampling_rate)
            else:
                audio_data, _ = librosa.load(
                    BytesIO(urlopen(url).read()),
                    sr=self.processor.feature_extractor.sampling_rate)
            audios.append(audio_data)
        return audios

    def audio_tower(self, audios, mask, stream, run_time=1):
        audios = audios.to(self.gpu_device)
        mask = mask.to(self.gpu_device)
        audio_inputs = {"input": audios.float(), "mask": mask}
        audio_output_info = self.session_audio.infer_shapes([
            TensorInfo("input", trt.DataType.FLOAT, audios.shape),
            TensorInfo("mask", trt.DataType.HALF, mask.shape)
        ])
        audio_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device=self.gpu_device)
            for t in audio_output_info
        }
        profiler.start("Audio")
        for _ in range(run_time):
            ok = self.session_audio.run(audio_inputs, audio_outputs,
                                        stream.cuda_stream)
        stream.synchronize()
        audio_time = profiler.stop("Audio") / run_time
        logger.info(f"TensorRT-LLM Audio latency: {audio_time:3f} sec ")

        assert ok, "Runtime execution failed for audio session"

        audio_features = audio_outputs["output"]

        return audio_features

    def generate_for_qwen_audio(
        self,
        input_tokens,
        args,
        prompt_table=None,
        extra_ids=None,
        run_time=1,
    ):
        input_ids = torch.as_tensor(input_tokens,
                                    device=self.gpu_device,
                                    dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device=self.gpu_device,
                                     dtype=torch.int32)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(args.max_new_tokens,
                             self.max_seq_len - max_input_length)

        prompt_table = prompt_table.unsqueeze(0)
        profiler.start("QWen")
        for _ in range(run_time):
            outputs = self.runner.generate(
                batch_input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=self.sampling_config.end_id,
                pad_id=self.sampling_config.pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=[[[151643], [151645]]],
                bad_words_list=self.sampling_config.bad_words_list,
                random_seed=args.random_seed,
                lora_uids=args.lora_task_uids,
                prompt_table=prompt_table,
                prompt_tasks="0",
                output_sequence_lengths=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                return_dict=True,
                return_all_generated_tokens=False,
                input_token_extra_ids=extra_ids)
            output_ids = outputs['output_ids']
            torch.cuda.synchronize()
        Qwen_time = profiler.stop("QWen") / run_time

        return output_ids, Qwen_time

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def qwen_infer(self,
                   input_text,
