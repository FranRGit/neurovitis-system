import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

// --- INTERFACES ACTUALIZADAS ---
export interface Diagnosis {
  disease: string;
  confidence: number;
  fuzzy_probs: { [key: string]: number }; 
}

export interface ExpertAnalysis {
  severity_label: string;
  tissue_damage_percent: number;
  roi_detected: boolean;
}

export interface Visuals {
  damage_mask_base64: string | null;
}

export interface NeuroResponse {
  status: 'success' | 'error'; // Campo clave
  message?: string;            // Opcional, solo viene si hay error
  code?: string;               // Código de error
  
  // Estos ahora son opcionales (?) porque si hay error, no vendrán
  diagnosis?: Diagnosis;
  expert_analysis?: ExpertAnalysis;
  visuals?: Visuals;
}


@Injectable({
  providedIn: 'root'
})
export class NeuroVitisService {

  private API_URL = environment.apiUrl; 

  constructor(private http: HttpClient) {}

  predict(imageFile: File): Observable<NeuroResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);
    return this.http.post<NeuroResponse>(`${this.API_URL}/predict`, formData);
  }

  sendMessageToAgent(message: string, contextData: any): Observable<any> {
    return this.http.post(`${this.API_URL}/chat`, { 
      message: message, 
      context: contextData 
    });
  }
  
  checkHealth(): Observable<any> {
    return this.http.get(`${this.API_URL}/health`);
  }
}