import { Component, ViewChild, ElementRef, OnInit, NgModule } from '@angular/core';
import { NeuroVitisService, NeuroResponse } from '../services/neurovitis.service';
import { CommonModule } from '@angular/common';
import { FormsModule, NgModel } from '@angular/forms';

interface ChatMessage {
  sender: 'user' | 'agent';
  text: string;
  time: string;
}

@Component({
  selector: 'app-analyzer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './analyzer.component.html',
  styleUrls: ['./analyzer.component.css']
})
export class AnalyzerComponent implements OnInit {
  // Variables de Estado
  selectedFile: File | null = null;
  imagePreview: string | null = null;
  result: NeuroResponse | null = null;
  isLoading = false;
  isCameraOpen = false;
  showImageModal = false;
  
  // NUEVA VARIABLE: Para manejar mensajes de error en la UI
  errorMessage: string | null = null;

  // Variables de Feedback
  feedbackStatus: 'pending' | 'validated' | 'corrected' = 'pending';
  showCorrectionDropdown = false;
  availableDiseases = ['Black Rot', 'Esca', 'Healthy', 'Leaf Blight'];
  selectedCorrection = '';

  // Variables de Chat
  isChatOpen = false;
  chatMessages: ChatMessage[] = [
    { sender: 'agent', text: 'Hola, soy NeuroBot. Sube una foto para comenzar.', time: this.getTime() }
  ];
  newMessage = '';

  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;

  constructor(private neuroService: NeuroVitisService) {}

  ngOnInit(): void {}

  // --- MÓDULO DE CARGA ---
  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  toggleCamera(): void {
    this.isCameraOpen = !this.isCameraOpen;
    if (this.isCameraOpen) {
      this.startCamera();
    } else {
      this.stopCamera();
    }
  }

  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      this.videoElement.nativeElement.srcObject = stream;
    } catch (err) {
      console.error('Error accediendo a la cámara:', err);
      alert('No se pudo acceder a la cámara');
    }
  }

  stopCamera() {
    const stream = this.videoElement?.nativeElement?.srcObject as MediaStream;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
  }

  captureImage() {
    const video = this.videoElement.nativeElement;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
      if (blob) {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        this.handleFile(file);
        this.toggleCamera(); // Cerrar cámara tras captura
      }
    });
  }

  private handleFile(file: File) {
    this.selectedFile = file;
    this.feedbackStatus = 'pending'; 
    this.showCorrectionDropdown = false;
    this.result = null; 
    
    // IMPORTANTE: Limpiar errores previos al cargar nueva imagen
    this.errorMessage = null;
    
    const reader = new FileReader();
    reader.onload = (e) => this.imagePreview = e.target?.result as string;
    reader.readAsDataURL(file);
  }

  // Métodos para controlar el modal
  openImageModal() {
    if (this.result?.visuals?.damage_mask_base64) {
      this.showImageModal = true;
    }
  }

  closeImageModal() {
    this.showImageModal = false;
  }

  // CONSUMO DE API 
  analyzeImage(): void {
    if (!this.selectedFile) return;

    this.isLoading = true;
    this.errorMessage = null; // Resetear mensaje de error
    this.result = null;       // Resetear resultado anterior

    this.neuroService.predict(this.selectedFile).subscribe({
      next: (res) => {
        this.isLoading = false;

        if (res.status === 'error') {
            this.errorMessage = res.message || 'Error desconocido al procesar la imagen.';
            this.addAgentMessage(`${this.errorMessage}`); // El bot también avisa
            return; // Detenemos aquí, no mostramos resultados
        }

        this.result = res;
        
        if (res.diagnosis) {
            this.addAgentMessage(`Diagnóstico completado: ${res.diagnosis.disease} (${res.diagnosis.confidence.toFixed(1)}%). ¿Necesitas recomendaciones?`);
        }
      },
      error: (err) => {
        console.error(err);
        this.isLoading = false;
        // Manejo de error de red (HTTP 500, 404, offline)
        this.errorMessage = 'Error de conexión con el servidor neuronal. Verifica que el backend esté activo.';
        this.addAgentMessage('Error de conexión con el servidor.');
      }
    });
  }

  //MÓDULO DE FEEDBACK (Cascarón)
  approveDiagnosis() {
    this.feedbackStatus = 'validated';
    console.log('Enviando a DB: Status VALIDATED');
  }

  enableCorrection() {
    this.showCorrectionDropdown = true;
  }

  submitCorrection() {
    if (this.selectedCorrection) {
      this.feedbackStatus = 'corrected';
      this.showCorrectionDropdown = false;
      console.log(`Enviando a DB: Status CORRECTED -> ${this.selectedCorrection}`);
    }
  }

  //MÓDULO DE CHAT (Cascarón)
  toggleChat() {
    this.isChatOpen = !this.isChatOpen;
  }

  sendMessage() {
    if (!this.newMessage.trim()) return;

    // 1. Mostrar mensaje en UI
    this.chatMessages.push({ sender: 'user', text: this.newMessage, time: this.getTime() });
    
    const msg = this.newMessage;
    this.newMessage = '';

    // 2. PREPARAR EL OBJETO DE CONTEXTO (Limpio, solo datos clave)
    let contextPayload = {};
    
    // Verificamos que tengamos un diagnóstico completo antes de enviarlo
    if (this.result && this.result.diagnosis && this.result.expert_analysis) {
        contextPayload = {
            disease: this.result.diagnosis.disease,
            confidence: this.result.diagnosis.confidence,
            fuzzy_probs: this.result.diagnosis.fuzzy_probs, 
            severity: this.result.expert_analysis.severity_label,
            damage_percent: this.result.expert_analysis.tissue_damage_percent,
            roi_detected: this.result.expert_analysis.roi_detected
        };
    }

    // 3. ENVIAR AL BACKEND
    this.neuroService.sendMessageToAgent(msg, contextPayload).subscribe({
      next: (res) => {
        if (res && res.response) {
            this.addAgentMessage(res.response);
        }
      },
      error: (err) => {
        this.addAgentMessage("Error conectando con NeuroBot.");
      }
    });
  }

  private addAgentMessage(text: string) {
    this.chatMessages.push({ sender: 'agent', text, time: this.getTime() });
  }

  private getTime(): string {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // --- HELPERS PARA LA VISTA ---
  getConfidenceColor(value: number): string {
    if (value > 80) return 'var(--neon-red)';     // Peligro/Certeza Alta (Enfermedad)
    if (value < 50) return 'var(--neon-yellow)';  // Incertidumbre
    return 'var(--neon-green)';                   // Standard
  }
  
  getAlertClass(): string {
    if (!this.result || !this.result.diagnosis) return '';
    
    const diagnosis = this.result.diagnosis;
    // Si es Healthy y alta confianza -> Verde
    if (diagnosis.disease === 'Healthy' && diagnosis.confidence > 80) return 'alert-safe';
    // Si es Enfermedad y alta confianza -> Rojo
    if (diagnosis.confidence > 80) return 'alert-danger';
    // Si es baja confianza -> Amarillo
    if (diagnosis.confidence < 50) return 'alert-warning';
    
    return 'alert-normal';
  }
  
  // Convierte el diccionario fuzzy_probs a array para iterar en HTML
  getDetailsArray() {
    if (!this.result || !this.result.diagnosis) return [];
    return Object.entries(this.result.diagnosis.fuzzy_probs).map(([key, value]) => ({ name: key, value }));
  }

  // --- NUEVO: Helper para Severidad ---
  getSeverityClass(label: string): string {
    if (!label) return '';
    const l = label.toLowerCase();
    
    if (l.includes('severo') || l.includes('crítico')) return 'severity-high';
    if (l.includes('moderado')) return 'severity-mid';
    if (l.includes('leve') || l.includes('sana') || l.includes('incierto')) return 'severity-low';
    
    return '';
  }
}