// Web Speech API wrapper for Diktao (configured in fr-FR)

export interface SpeechRecognitionResultState {
  isListening: boolean;
  transcript: string;
  error: string | null;
  isSupported: boolean;
}

// Window typing for speech recognition
interface IWindow extends Window {
  SpeechRecognition?: any;
  webkitSpeechRecognition?: any;
}

export class DiktaoSpeechRecognizer {
  private recognition: any = null;
  private isListening = false;
  private isStarting = false;
  private onTranscriptChange?: (text: string, isFinal: boolean) => void;
  private onErrorChange?: (errorMessage: string) => void;
  private onStatusChange?: (listening: boolean) => void;
  private accumulatedText = '';

  constructor(
    onTranscript: (text: string, isFinal: boolean) => void,
    onError: (errorMessage: string) => void,
    onStatus: (listening: boolean) => void
  ) {
    this.onTranscriptChange = onTranscript;
    this.onErrorChange = onError;
    this.onStatusChange = onStatus;
    this.init();
  }

  private init() {
    const win = window as IWindow;
    const SpeechRecognitionClass = win.SpeechRecognition || win.webkitSpeechRecognition;

    if (!SpeechRecognitionClass) {
      this.onErrorChange?.(
        "Votre navigateur ne supporte pas la reconnaissance vocale native. Vous pouvez taper votre texte directement au clavier."
      );
      return;
    }

    try {
      this.recognition = new SpeechRecognitionClass();
      this.recognition.lang = 'fr-FR';
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 1;

      this.recognition.onstart = () => {
        this.isStarting = false;
        this.isListening = true;
        this.onStatusChange?.(true);
      };

      this.recognition.onresult = (event: any) => {
        let interim = '';
        let final = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const res = event.results[i];
          if (res.isFinal) {
            final += res[0].transcript + ' ';
          } else {
            interim += res[0].transcript;
          }
        }

        if (final) {
          this.accumulatedText += final;
        }

        const fullDisplay = (this.accumulatedText + interim).trim();
        this.onTranscriptChange?.(fullDisplay, Boolean(final));
      };

      this.recognition.onerror = (event: any) => {
        this.isStarting = false;
        console.warn('Speech recognition event error:', event.error);
        if (event.error === 'not-allowed') {
          this.onErrorChange?.(
            "L'accès au micro a été refusé. Veuillez autoriser le microphone dans les paramètres de votre navigateur ou taper votre texte."
          );
        } else if (event.error === 'no-speech') {
          // Silent timeout, non blocking
        } else if (event.error === 'network') {
          this.onErrorChange?.(
            "Problème réseau lors de la reconnaissance vocale. Vous pouvez continuer ou taper le texte manuellement."
          );
        } else {
          this.onErrorChange?.(`Note vocale : ${event.error}. Vous pouvez aussi saisir le texte.`);
        }
        this.stop();
      };

      this.recognition.onend = () => {
        this.isStarting = false;
        this.isListening = false;
        this.onStatusChange?.(false);
      };
    } catch (err) {
      this.isStarting = false;
      console.warn('SpeechRecognition initialization error', err);
      this.onErrorChange?.("Erreur lors de l'activation du microphone.");
    }
  }

  public start() {
    if (!this.recognition) {
      this.init();
    }
    if (!this.recognition) return;

    // Guard against starting if already active or pending start
    if (this.isListening || this.isStarting) {
      return;
    }

    this.isStarting = true;

    try {
      this.recognition.start();
    } catch (e: any) {
      this.isStarting = false;
      // Handle InvalidStateError safely
      if (e?.name === 'InvalidStateError' || String(e).includes('already started')) {
        console.warn('Speech recognition already active, updating internal state');
        this.isListening = true;
        this.onStatusChange?.(true);
      } else {
        console.warn('Failed to start speech recognition safely:', e);
      }
    }
  }

  public stop() {
    this.isStarting = false;
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (err) {
        try {
          this.recognition.abort();
        } catch {}
      }
    }
    this.isListening = false;
    this.onStatusChange?.(false);
  }

  public toggle() {
    if (this.isListening || this.isStarting) {
      this.stop();
    } else {
      this.start();
    }
  }

  public clear() {
    this.accumulatedText = '';
    this.onTranscriptChange?.('', true);
  }

  public setText(text: string) {
    this.accumulatedText = text;
    this.onTranscriptChange?.(text, true);
  }

  public getIsListening(): boolean {
    return this.isListening || this.isStarting;
  }
}
