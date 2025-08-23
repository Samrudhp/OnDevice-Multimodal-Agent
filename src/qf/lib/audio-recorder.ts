export interface AudioSample {
  base64: string;
  duration: number; // seconds
  sampleRate: number;
}

export async function recordShortAudio(durationMs: number = 1500, sampleRate: number = 16000): Promise<AudioSample> {
  // Use dynamic imports so the app can still bundle if expo-av isn't installed.
  let Audio: any;
  let FileSystem: any;
  try {
    // Use runtime require via eval to avoid Metro static analysis resolving optional deps
    const req: any = eval('require');
    const audioMod = req('expo-av');
    Audio = audioMod.Audio;
    FileSystem = req('expo-file-system');
  } catch (err) {
    throw new Error('Missing dependency: please install expo-av and expo-file-system in the client app (npm install expo-av expo-file-system)');
  }

  // Request permissions
  const { status } = await Audio.requestPermissionsAsync();
  if (status !== 'granted') throw new Error('Microphone permission not granted');

  // Configure audio mode for recording
  await Audio.setAudioModeAsync({ allowsRecordingIOS: true, interruptionModeIOS: Audio.INTERRUPTION_MODE_IOS_DO_NOT_MIX, playsInSilentModeIOS: true, staysActiveInBackground: false, shouldDuckAndroid: true });

  const recording = new Audio.Recording();
  try {
    await recording.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
    await recording.startAsync();

    await new Promise(resolve => setTimeout(resolve, Math.max(200, durationMs)));

    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    if (!uri) throw new Error('Recording failed, no file URI');

    const base64 = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
    return { base64, duration: durationMs / 1000, sampleRate };
  } catch (err) {
    try { await recording.stopAndUnloadAsync(); } catch (e) { /* ignore */ }
    throw err;
  }
}

export default recordShortAudio;
