import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

// Interfaces para tipar la respuesta (Buenas prácticas)
export interface NeuroResponse {
  confidence: number;
  details: {
    "Black Rot": number;
    "Esca": number;
    "Healthy": number;
    "Leaf Blight": number;
  };
  diagnosis: string;
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