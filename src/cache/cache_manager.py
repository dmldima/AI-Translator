def get_batch(self, lookup_data: List[Dict[str, str]]) -> List[Optional['CacheEntry']]:
        """
        OPTIMIZATION: Batch cache lookup with single SQL query.
        
        Args:
            lookup_data: List of dicts with keys: source_text, source_lang, 
                        target_lang, glossary_version, domain
        
        Returns:
            List of CacheEntry or None (same order as input)
        """
        if not lookup_data:
            return []
        
        try:
            # Generate keys for all requests
            keys = []
            for data in lookup_data:
                key = self.generate_key(
                    data['source_text'],
                    data['source_lang'],
                    data['target_lang'],
                    data.get('glossary_version', ''),
                    data.get('domain', '')
                )
                keys.append(key)
            
            # Batch SQL query
            with self.storage._get_connection() as conn:
                placeholders = ','.join('?' * len(keys))
                query = f"SELECT * FROM cache WHERE key IN ({placeholders})"
                cur = conn.execute(query, keys)
                
                # Build results map
                results_map = {}
                for row in cur.fetchall():
                    from ..cache.cache_manager import CacheEntry
                    entry = CacheEntry(**dict(row))
                    results_map[row['key']] = entry
            
            # Return in original order
            results = []
            for key, data in zip(keys, lookup_data):
                entry = results_map.get(key)
                
                # Validate glossary version and staleness
                if entry:
                    if entry.glossary_version != data.get('glossary_version', ''):
                        entry = None
                    elif self._is_stale(entry.timestamp):
                        entry = None
                
                results.append(entry)
            
            # Update stats
            hits = sum(1 for r in results if r is not None)
            misses = len(results) - hits
            
            logger.info(
                f"Batch cache lookup: {hits}/{len(lookup_data)} hits "
                f"({hits/len(lookup_data)*100:.1f}% hit rate)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch cache lookup error: {e}")
            # Fallback to individual lookups
            return [
                self.get(
                    data['source_text'],
                    data['source_lang'],
                    data['target_lang'],
                    data.get('glossary_version', ''),
                    data.get('domain', '')
                )
                for data in lookup_data
            ]
