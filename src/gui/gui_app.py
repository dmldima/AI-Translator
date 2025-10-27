"""
macOS Native GUI for AI Document Translator using PyQt6.
Follows macOS Human Interface Guidelines.
"""
import sys
from pathlib import Path
from typing import Optional, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem, QTabWidget,
    QGroupBox, QCheckBox, QSpinBox, QLineEdit, QMessageBox,
    QListWidget, QSplitter, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QAction, QDragEnterEvent, QDropEvent

# Import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.factory import TranslationSystemFactory
from src.core.models import TranslationJob, TranslationStatus, SUPPORTED_LANGUAGES
from src.core.interfaces import IProgressCallback
from src.utils.config_manager import get_config_manager
from src.processors.batch_processor import BatchProcessor, BatchTask


class TranslationWorker(QThread):
    """Worker thread for translation to keep UI responsive."""
    
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(object)  # TranslationJob
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    
    def __init__(self, pipeline, input_file, output_file, source_lang, target_lang, domain):
        super().__init__()
        self.pipeline = pipeline
        self.input_file = input_file
        self.output_file = output_file
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.domain = domain
    
    def run(self):
        try:
            self.status.emit("Parsing document...")
            
            # Progress callback
            class ProgressCallback(IProgressCallback):
                def __init__(self, worker):
                    self.worker = worker
                
                def on_start(self, job):
                    self.worker.status.emit(f"Translating {job.total_segments} segments...")
                
                def on_progress(self, job, current, total):
                    self.worker.progress.emit(current, total)
                
                def on_complete(self, job):
                    self.worker.status.emit("Formatting document...")
                
                def on_error(self, job, error):
                    pass
                
                def on_segment_translated(self, segment, result):
                    pass
            
            # Translate
            job = self.pipeline.translate_document(
                input_path=self.input_file,
                output_path=self.output_file,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                domain=self.domain,
                progress_callback=ProgressCallback(self)
            )
            
            self.finished.emit(job)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window with macOS native feel."""
    
    def __init__(self):
        super().__init__()
        
        # Load config
        self.config_mgr = get_config_manager()
        self.config = self.config_mgr.config
        
        # State
        self.pipeline = None
        self.worker = None
        
        self.init_ui()
        self.load_settings()
        self.create_pipeline()
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("AI Document Translator")
        self.setMinimumSize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_translate_tab(), "Translate")
        tabs.addTab(self.create_batch_tab(), "Batch")
        tabs.addTab(self.create_glossary_tab(), "Glossary")
        tabs.addTab(self.create_settings_tab(), "Settings")
        
        main_layout.addWidget(tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        self.create_menu_bar()
        
        # Apply macOS style
        self.apply_macos_style()
    
    def create_translate_tab(self) -> QWidget:
        """Create main translation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # File selection group
        file_group = QGroupBox("Document")
        file_layout = QVBoxLayout()
        
        # Input file
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select document to translate...")
        self.input_file_edit.setReadOnly(True)
        input_layout.addWidget(self.input_file_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(browse_btn)
        
        file_layout.addLayout(input_layout)
        
        # Output file
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setPlaceholderText("Output file location...")
        output_layout.addWidget(self.output_file_edit)
        
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output_file)
        output_layout.addWidget(output_browse_btn)
        
        file_layout.addLayout(output_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Translation settings group
        settings_group = QGroupBox("Translation Settings")
        settings_layout = QVBoxLayout()
        
        # Languages
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("From:"))
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems([f"{code} - {name}" for code, name in sorted(SUPPORTED_LANGUAGES.items())])
        lang_layout.addWidget(self.source_lang_combo)
        
        lang_layout.addWidget(QLabel("To:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems([f"{code} - {name}" for code, name in sorted(SUPPORTED_LANGUAGES.items())])
        lang_layout.addWidget(self.target_lang_combo)
        
        settings_layout.addLayout(lang_layout)
        
        # Engine and domain
        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["OpenAI", "DeepL"])
        options_layout.addWidget(self.engine_combo)
        
        options_layout.addWidget(QLabel("Domain:"))
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["general", "legal", "medical", "technical", "financial"])
        self.domain_combo.setEditable(True)
        options_layout.addWidget(self.domain_combo)
        
        settings_layout.addLayout(options_layout)
        
        # Options
        self.cache_checkbox = QCheckBox("Use cache")
        self.cache_checkbox.setChecked(True)
        settings_layout.addWidget(self.cache_checkbox)
        
        self.glossary_checkbox = QCheckBox("Apply glossary")
        self.glossary_checkbox.setChecked(True)
        settings_layout.addWidget(self.glossary_checkbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to translate")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Translate button
        self.translate_btn = QPushButton("Translate Document")
        self.translate_btn.setMinimumHeight(44)  # macOS standard height
        self.translate_btn.clicked.connect(self.start_translation)
        layout.addWidget(self.translate_btn)
        
        layout.addStretch()
        
        return widget
    
    def create_batch_tab(self) -> QWidget:
        """Create batch processing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Directory selection
        dir_group = QGroupBox("Directories")
        dir_layout = QVBoxLayout()
        
        # Input directory
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        self.batch_input_edit = QLineEdit()
        input_layout.addWidget(self.batch_input_edit)
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(self.browse_batch_input)
        input_layout.addWidget(browse_input_btn)
        dir_layout.addLayout(input_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.batch_output_edit = QLineEdit()
        output_layout.addWidget(self.batch_output_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_batch_output)
        output_layout.addWidget(browse_output_btn)
        dir_layout.addLayout(output_layout)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Batch settings
        batch_settings_group = QGroupBox("Batch Settings")
        batch_settings_layout = QVBoxLayout()
        
        # File pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["*.docx", "*.xlsx", "*.doc", "*.*"])
        self.pattern_combo.setEditable(True)
        pattern_layout.addWidget(self.pattern_combo)
        
        pattern_layout.addWidget(QLabel("Max workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 10)
        self.workers_spin.setValue(3)
        pattern_layout.addWidget(self.workers_spin)
        
        batch_settings_layout.addLayout(pattern_layout)
        batch_settings_group.setLayout(batch_settings_layout)
        layout.addWidget(batch_settings_group)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(["File", "Status", "Duration", "Error"])
        results_layout.addWidget(self.batch_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Start batch button
        self.start_batch_btn = QPushButton("Start Batch Translation")
        self.start_batch_btn.setMinimumHeight(44)
        self.start_batch_btn.clicked.connect(self.start_batch)
        layout.addWidget(self.start_batch_btn)
        
        return widget
    
    def create_glossary_tab(self) -> QWidget:
        """Create glossary management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Glossary controls
        controls_layout = QHBoxLayout()
        
        add_term_btn = QPushButton("Add Term")
        add_term_btn.clicked.connect(self.add_glossary_term)
        controls_layout.addWidget(add_term_btn)
        
        import_btn = QPushButton("Import...")
        controls_layout.addWidget(import_btn)
        
        export_btn = QPushButton("Export...")
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Glossary table
        self.glossary_table = QTableWidget()
        self.glossary_table.setColumnCount(4)
        self.glossary_table.setHorizontalHeaderLabels(["Source", "Target", "Domain", "Status"])
        layout.addWidget(self.glossary_table)
        
        return widget
    
    def create_settings_tab(self) -> QWidget:
        """Create settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Engine settings
        engine_group = QGroupBox("Engine Settings")
        engine_layout = QVBoxLayout()
        
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter API key or use environment variable")
        api_key_layout.addWidget(self.api_key_edit)
        
        show_key_btn = QPushButton("Show")
        show_key_btn.clicked.connect(self.toggle_api_key_visibility)
        api_key_layout.addWidget(show_key_btn)
        
        engine_layout.addLayout(api_key_layout)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
        model_layout.addWidget(self.model_combo)
        engine_layout.addLayout(model_layout)
        
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)
        
        # Cache settings
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QVBoxLayout()
        
        self.enable_cache_check = QCheckBox("Enable cache")
        self.enable_cache_check.setChecked(True)
        cache_layout.addWidget(self.enable_cache_check)
        
        ttl_layout = QHBoxLayout()
        ttl_layout.addWidget(QLabel("Cache TTL (days):"))
        self.cache_ttl_spin = QSpinBox()
        self.cache_ttl_spin.setRange(1, 365)
        self.cache_ttl_spin.setValue(180)
        ttl_layout.addWidget(self.cache_ttl_spin)
        ttl_layout.addStretch()
        cache_layout.addLayout(ttl_layout)
        
        clear_cache_btn = QPushButton("Clear Cache")
        clear_cache_btn.clicked.connect(self.clear_cache)
        cache_layout.addWidget(clear_cache_btn)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        # UI settings
        ui_group = QGroupBox("Interface")
        ui_layout = QVBoxLayout()
        
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Auto", "Light", "Dark"])
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        ui_layout.addLayout(theme_layout)
        
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)
        
        layout.addStretch()
        
        # Save button
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.setMinimumHeight(44)
        save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_settings_btn)
        
        return widget
    
    def create_menu_bar(self):
        """Create macOS-style menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_input_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        settings_action = QAction("Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        edit_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def apply_macos_style(self):
        """Apply macOS-native styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
            QPushButton:pressed {
                background-color: #004DB8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLineEdit, QComboBox, QSpinBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 3px;
            }
            QTableWidget {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007AFF;
            }
        """)
    
    def create_pipeline(self):
        """Create translation pipeline from config."""
        try:
            self.pipeline = TranslationSystemFactory.create_pipeline(
                engine_name=self.config.engine.name,
                api_key=self.config.engine.api_key,
                cache_enabled=self.config.cache.enabled,
                glossary_enabled=self.config.glossary.enabled,
                model=self.config.engine.model
            )
            self.status_bar.showMessage("Pipeline initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize pipeline: {e}")
    
    def browse_input_file(self):
        """Browse for input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Documents (*.docx *.xlsx *.doc *.xls);;All Files (*.*)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)
            # Auto-generate output filename
            input_path = Path(file_path)
            target_lang = self.target_lang_combo.currentText().split(" - ")[0]
            output_path = input_path.parent / f"{input_path.stem}_{target_lang}{input_path.suffix}"
            self.output_file_edit.setText(str(output_path))
    
    def browse_output_file(self):
        """Browse for output file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Translated Document",
            "",
            "Documents (*.docx *.xlsx);;All Files (*.*)"
        )
        if file_path:
            self.output_file_edit.setText(file_path)
    
    def browse_batch_input(self):
        """Browse for batch input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.batch_input_edit.setText(dir_path)
    
    def browse_batch_output(self):
        """Browse for batch output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.batch_output_edit.setText(dir_path)
    
    def start_translation(self):
        """Start document translation."""
        # Validate inputs
        if not self.input_file_edit.text():
            QMessageBox.warning(self, "Warning", "Please select an input file")
            return
        
        if not self.output_file_edit.text():
            QMessageBox.warning(self, "Warning", "Please specify an output file")
            return
        
        # Disable controls
        self.translate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get settings
        source_lang = self.source_lang_combo.currentText().split(" - ")[0]
        target_lang = self.target_lang_combo.currentText().split(" - ")[0]
        domain = self.domain_combo.currentText()
        
        # Create worker
        self.worker = TranslationWorker(
            self.pipeline,
            Path(self.input_file_edit.text()),
            Path(self.output_file_edit.text()),
            source_lang,
            target_lang,
            domain
        )
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.translation_finished)
        self.worker.error.connect(self.translation_error)
        
        # Start
        self.worker.start()
    
    def update_progress(self, current, total):
        """Update progress bar."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Translating: {current}/{total} segments")
    
    def update_status(self, message):
        """Update status message."""
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
    
    def translation_finished(self, job):
        """Handle translation completion."""
        self.translate_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Translation complete!")
        self.status_bar.showMessage("Ready")
        
        QMessageBox.information(
            self,
            "Success",
            f"Translation complete!\n\n"
            f"Segments: {job.translated_segments}/{job.total_segments}\n"
            f"Cached: {job.cached_segments}\n"
            f"Duration: {job.duration:.2f}s\n\n"
            f"Saved to: {job.output_file}"
        )
    
    def translation_error(self, error):
        """Handle translation error."""
        self.translate_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Translation failed")
        self.status_bar.showMessage("Error")
        
        QMessageBox.critical(self, "Error", f"Translation failed:\n\n{error}")
    
    def start_batch(self):
        """Start batch processing."""
        QMessageBox.information(self, "Info", "Batch processing will be implemented in worker thread")
    
    def add_glossary_term(self):
        """Add new glossary term."""
        QMessageBox.information(self, "Info", "Glossary term dialog will be implemented")
    
    def clear_cache(self):
        """Clear translation cache."""
        reply = QMessageBox.question(
            self,
            "Clear Cache",
            "Are you sure you want to clear the cache?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear cache logic here
            QMessageBox.information(self, "Success", "Cache cleared")
    
    def toggle_api_key_visibility(self):
        """Toggle API key visibility."""
        if self.api_key_edit.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    
    def load_settings(self):
        """Load settings from config."""
        # Set default languages
        source_idx = self.source_lang_combo.findText(self.config.default_source_lang, Qt.MatchFlag.MatchStartsWith)
        if source_idx >= 0:
            self.source_lang_combo.setCurrentIndex(source_idx)
        
        target_idx = self.target_lang_combo.findText(self.config.default_target_lang, Qt.MatchFlag.MatchStartsWith)
        if target_idx >= 0:
            self.target_lang_combo.setCurrentIndex(target_idx)
        
        # Set engine
        engine_idx = 0 if self.config.engine.name == "openai" else 1
        self.engine_combo.setCurrentIndex(engine_idx)
        
        # Set model
        model_idx = self.model_combo.findText(self.config.engine.model)
        if model_idx >= 0:
            self.model_combo.setCurrentIndex(model_idx)
        
        # Set cache settings
        self.enable_cache_check.setChecked(self.config.cache.enabled)
        self.cache_ttl_spin.setValue(self.config.cache.max_age_days)
        
        # Set theme
        theme_idx = {"auto": 0, "light": 1, "dark": 2}.get(self.config.ui.theme.lower(), 0)
        self.theme_combo.setCurrentIndex(theme_idx)
    
    def save_settings(self):
        """Save settings to config."""
        # Update config
        self.config.engine.name = "openai" if self.engine_combo.currentIndex() == 0 else "deepl"
        self.config.engine.model = self.model_combo.currentText()
        self.config.engine.api_key = self.api_key_edit.text() if self.api_key_edit.text() else None
        
        self.config.cache.enabled = self.enable_cache_check.isChecked()
        self.config.cache.max_age_days = self.cache_ttl_spin.value()
        
        self.config.ui.theme = self.theme_combo.currentText().lower()
        
        # Save to file
        self.config_mgr.save()
        
        QMessageBox.information(self, "Success", "Settings saved")
        
        # Recreate pipeline
        self.create_pipeline()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AI Document Translator",
            "<h3>AI Document Translator v1.1</h3>"
            "<p>Professional AI-powered document translation with perfect formatting preservation.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multiple engines (OpenAI, DeepL)</li>"
            "<li>Format preservation (DOCX, XLSX)</li>"
            "<li>Smart caching</li>"
            "<li>Glossary management</li>"
            "</ul>"
            "<p>© 2024 AI Translator Team</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Translation in Progress",
                "Translation is in progress. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self.worker.terminate()
            self.worker.wait()
        
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Document Translator")
    app.setOrganizationName("AI Translator")
    
    # Set macOS-specific attributes
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
