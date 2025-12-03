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
  status: string;
  diagnosis: Diagnosis;
  expert_analysis: ExpertAnalysis;
  visuals: Visuals;
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
  
  checkHealth(): Observable<any> {
    return this.http.get(`${this.API_URL}/health`);
  }
}