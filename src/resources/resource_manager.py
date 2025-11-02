def optimize_indices(self, table_name: str, index_columns: List[str]):
        """
        OPTIMIZATION: Create optimal indices for cache/glossary tables.
        
        Args:
            table_name: Table name
            index_columns: List of columns to index
        """
        try:
            conn = self.get_connection()
            
            for col in index_columns:
                index_name = f"idx_{table_name}_{col}"
                
                # Check if index exists
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (index_name,)
                )
                
                if not cur.fetchone():
                    # Create index
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table_name}({col})"
                    )
                    logger.info(f"Created index: {index_name}")
            
            # Analyze for query optimization
            conn.execute(f"ANALYZE {table_name}")
            
            logger.info(f"Optimized indices for {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to optimize indices: {e}")
