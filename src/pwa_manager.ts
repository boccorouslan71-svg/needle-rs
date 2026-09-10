// PWA Manager for Diktao

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform: string }>;
}

export class PWAManager {
  private deferredPrompt: BeforeInstallPromptEvent | null = null;
  private isStandalone = false;
  private isIOS = false;
  private onInstallableChange?: (canInstall: boolean) => void;

  constructor(onInstallable?: (canInstall: boolean) => void) {
    this.onInstallableChange = onInstallable;
    this.init();
  }

  private init() {
    // Check if running as installed standalone PWA
    const nav = window.navigator as any;
    this.isStandalone = 
      window.matchMedia('(display-mode: standalone)').matches ||
      nav.standalone === true;

    // Detect iOS
    const ua = window.navigator.userAgent.toLowerCase();
    this.isIOS = /iphone|ipad|ipod/.test(ua);

    // Register Service Worker
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(
          (reg) => {
            console.log('Diktao Service Worker registered with scope:', reg.scope);
          },
          (err) => {
            console.warn('Service Worker registration failed:', err);
          }
        );
      });
    }

    // Capture install prompt event
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this.deferredPrompt = e as BeforeInstallPromptEvent;
      this.onInstallableChange?.(true);
    });

    window.addEventListener('appinstalled', () => {
      this.isStandalone = true;
      this.deferredPrompt = null;
      this.onInstallableChange?.(false);
      console.log('Diktao PWA installed successfully!');
    });
  }

  public async promptInstall(): Promise<boolean> {
    if (!this.deferredPrompt) return false;
    await this.deferredPrompt.prompt();
    const { outcome } = await this.deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      this.isStandalone = true;
      this.deferredPrompt = null;
      this.onInstallableChange?.(false);
      return true;
    }
    return false;
  }

  public getIsInstallable(): boolean {
    return !!this.deferredPrompt;
  }

  public getIsStandalone(): boolean {
    return this.isStandalone;
  }

  public getIsIOS(): boolean {
    return this.isIOS;
  }
}
