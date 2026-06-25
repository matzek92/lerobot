// Liest einen verstellbaren Widerstand zwischen A0 und GND aus
// ueber den internen Pullup-Widerstand des Arduino.
// Ausgabe: ADC-Wert, Spannung und grob geschaetzter Widerstand.

const int analogPin1 = A0;
const int analogPin2 = A1;

// Interner Pullup ist ungenau (typisch ca. 20k..50k).
// Fuer eine grobe Schaetzung nehmen wir hier 30000 Ohm an.
const float R_PULLUP_OHM = 30000.0;

// Anzahl Werte fuer gleitenden Mittelwert
const int WINDOW_SIZE = 10;

// 1 kHz Messrate = 1 Messung pro Millisekunde
const unsigned long SAMPLE_INTERVAL_MS = 1;

unsigned long lastSampleMs = 0;
float window1[WINDOW_SIZE];
float window2[WINDOW_SIZE];
int windowCount = 0;
int windowIndex = 0;
float windowSum1 = 0.0;
float windowSum2 = 0.0;
float prevMean1 = 0.0;
float prevMean2 = 0.0;
bool hasPrevMean = false;

void setup() {
  Serial.begin(115200);
  pinMode(analogPin1, INPUT_PULLUP);
  pinMode(analogPin2, INPUT_PULLUP);
  delay(200);
  for (int i = 0; i < WINDOW_SIZE; i++) {
    window1[i] = 0.0;
    window2[i] = 0.0;
  }
  Serial.println("Start: 1kHz Messung an A0 und A1 gegen GND");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  float liveRaw1 = (float)analogRead(analogPin1);  // 0..1023
  float liveRaw2 = (float)analogRead(analogPin2);  // 0..1023

  if (windowCount < WINDOW_SIZE) {
    window1[windowIndex] = liveRaw1;
    window2[windowIndex] = liveRaw2;
    windowSum1 += liveRaw1;
    windowSum2 += liveRaw2;
    windowCount++;
  } else {
    windowSum1 -= window1[windowIndex];
    windowSum2 -= window2[windowIndex];
    window1[windowIndex] = liveRaw1;
    window2[windowIndex] = liveRaw2;
    windowSum1 += liveRaw1;
    windowSum2 += liveRaw2;
  }

  windowIndex = (windowIndex + 1) % WINDOW_SIZE;

  float meanRaw1 = windowSum1 / (float)windowCount;
  float meanRaw2 = windowSum2 / (float)windowCount;
  float deltaMeanRaw1 = hasPrevMean ? (meanRaw1 - prevMean1) : 0.0;
  float deltaMeanRaw2 = hasPrevMean ? (meanRaw2 - prevMean2) : 0.0;
  prevMean1 = meanRaw1;
  prevMean2 = meanRaw2;
  hasPrevMean = true;

  float liveVoltage1 = liveRaw1 * (5.0 / 1023.0);   // bei 5V-Board
  float liveVoltage2 = liveRaw2 * (5.0 / 1023.0);   // bei 5V-Board
  float meanVoltage1 = meanRaw1 * (5.0 / 1023.0);
  float meanVoltage2 = meanRaw2 * (5.0 / 1023.0);

  // Rvar = Rpullup * raw / (1023 - raw)
  float rVar1 = -1.0;
  float rVar2 = -1.0;
  if (meanRaw1 < 1023.0) {
    rVar1 = R_PULLUP_OHM * meanRaw1 / (1023.0 - meanRaw1);
  }
  if (meanRaw2 < 1023.0) {
    rVar2 = R_PULLUP_OHM * meanRaw2 / (1023.0 - meanRaw2);
  }

  Serial.print("A0 ADC_live: ");
  Serial.print(liveRaw1, 1);
  Serial.print(" | ADC_mean10: ");
  Serial.print(meanRaw1, 1);
  Serial.print(" | dMean: ");
  Serial.print(deltaMeanRaw1, 2);
  Serial.print(" | U_live: ");
  Serial.print(liveVoltage1, 3);
  Serial.print(" V | U_mean10: ");
  Serial.print(meanVoltage1, 3);
  Serial.print(" V | R~(mean): ");

  if (rVar1 >= 0.0 && rVar1 < 1e7) {
    Serial.print(rVar1, 0);
    Serial.print(" Ohm");
  } else {
    Serial.print("unendlich / ausser Bereich");
  }

  Serial.print(" || A1 ADC_live: ");
  Serial.print(liveRaw2, 1);
  Serial.print(" | ADC_mean10: ");
  Serial.print(meanRaw2, 1);
  Serial.print(" | dMean: ");
  Serial.print(deltaMeanRaw2, 2);
  Serial.print(" | U_live: ");
  Serial.print(liveVoltage2, 3);
  Serial.print(" V | U_mean10: ");
  Serial.print(meanVoltage2, 3);
  Serial.print(" V | R~(mean): ");

  if (rVar2 >= 0.0 && rVar2 < 1e7) {
    Serial.print(rVar2, 0);
    Serial.println(" Ohm");
  } else {
    Serial.println("unendlich / ausser Bereich");
  }
}
