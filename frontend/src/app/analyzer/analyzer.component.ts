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
    this.neuroService.predict(this.selectedFile).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
        
        // Trigger automático para el chat
        // Actualizado para acceder a res.diagnosis.disease
        this.addAgentMessage(`Diagnóstico completado: ${res.diagnosis.disease} (${res.diagnosis.confidence.toFixed(1)}%). ¿Necesitas recomendaciones?`);
      },
      error: (err) => {
        console.error(err);
        alert('Error conectando con el servidor neuronal');
        this.isLoading = false;
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

    // 1. Agregar mensaje usuario
    this.chatMessages.push({ sender: 'user', text: this.newMessage, time: this.getTime() });
    
    // 2. Preparar contexto actualizado
    const context = this.result ? 
      `Contexto: Diagnóstico ${this.result.diagnosis.disease}, Confianza ${this.result.diagnosis.confidence}%, Severidad: ${this.result.expert_analysis.severity_label}` : 
      'Contexto: Sin diagnóstico activo';
    
    console.log(`Enviando a Agente LLM: "${this.newMessage}" + [${context}]`);

    // 3. Simular respuesta del agente
    setTimeout(() => {
      this.addAgentMessage("Entendido. Como modelo de lenguaje (simulado), te recomiendo verificar la humedad del suelo.");
    }, 1000);

    this.newMessage = '';
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
    if (!this.result) return '';
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
    if (!this.result) return [];
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