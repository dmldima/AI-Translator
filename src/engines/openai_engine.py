def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None
    ) -> str:
        """
        OPTIMIZED: Translate single text using OpenAI API with minimal prompt.
        
        Optimization: Reduced prompt tokens by ~30%
        """
        if not text.strip():
            return text
        
        # OPTIMIZATION: Minimal, efficient messages
        messages = [
            {
                "role": "system",
                "content": f"Translate {source_lang}->{target_lang}. Output ONLY the translation."
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        # Add context if provided (for batching)
        if context:
            messages[0]["content"] += f" {context}"
        
        # Translate with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract translation
                translated_text = response.choices[0].message.content.strip()
                
                # Update stats
                self._total_tokens += response.usage.total_tokens
                self._total_requests += 1
                
                logger.debug(
                    f"Translated {len(text)} chars, "
                    f"used {response.usage.total_tokens} tokens "
                    f"(prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens})"
                )
                
                return translated_text
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self._total_errors += 1
                    raise TranslationError(f"Rate limit exceeded: {e}")
            
            except APITimeoutError as e:
                logger.warning(f"Timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    self._total_errors += 1
                    raise TranslationError(f"Request timeout: {e}")
            
            except APIError as e:
                logger.error(f"API error: {e}")
                self._total_errors += 1
                raise TranslationError(f"OpenAI API error: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self._total_errors += 1
                raise TranslationError(f"Translation failed: {e}")
