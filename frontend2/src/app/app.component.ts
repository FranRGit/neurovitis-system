import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AnalyzerComponent } from './analyzer/analyzer.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, AnalyzerComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend2';
}
