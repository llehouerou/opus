// Copyright © Go Opus Authors (see AUTHORS file)
//
// License for use of this code is detailed in the LICENSE file

package opus

import (
	"sync"
	"testing"
)

// The Wasm runtime, module and the cached api.Function handles live in a single
// process-wide wasmContext, so every Encoder/Decoder shares them. wazero
// documents api.Function.Call as "not goroutine-safe (...) this should not be
// called multiple times until the previous Call returns", which means two
// encoders running on separate goroutines reuse one callEngine and corrupt its
// Wasm stack. That surfaces as an unrecoverable runtime failure ("split stack
// overflow", "unsafe.Slice: len out of range", "unexpected return pc") rather
// than a returned error, so it takes the whole process down.
//
// These tests drive independent encoders/decoders concurrently, which is the
// natural usage when serving several audio streams from one process.

const (
	concurrentSampleRate = 48000
	concurrentChannels   = 1
	// 960 samples = 20ms @ 48kHz, the frame size Opus expects for real-time audio.
	concurrentFrameSize = 960
)

// pcmTone builds a deterministic non-silent frame. Silence compresses to a couple
// of bytes and barely exercises the encoder, so use a simple ramp instead.
func pcmTone(frameSize int, seed int) []int16 {
	pcm := make([]int16, frameSize)
	for i := range pcm {
		pcm[i] = int16(((i + seed) % 256) * 64)
	}
	return pcm
}

func TestEncoderConcurrent(t *testing.T) {
	const (
		encoders         = 4
		framesPerEncoder = 250
	)

	var wg sync.WaitGroup
	errs := make(chan error, encoders)

	for e := 0; e < encoders; e++ {
		enc, err := NewEncoder(concurrentSampleRate, concurrentChannels, AppAudio)
		if err != nil {
			t.Fatalf("encoder %d: failed to create: %v", e, err)
		}

		wg.Add(1)
		go func(enc *Encoder, seed int) {
			defer wg.Done()
			pcm := pcmTone(concurrentFrameSize, seed)
			out := make([]byte, 4000)
			for i := 0; i < framesPerEncoder; i++ {
				n, err := enc.Encode(pcm, out)
				if err != nil {
					errs <- err
					return
				}
				if n <= 0 {
					errs <- errEncUninitialized
					return
				}
			}
		}(enc, e*37)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent encode failed: %v", err)
	}
}

func TestDecoderConcurrent(t *testing.T) {
	const (
		decoders         = 4
		framesPerDecoder = 250
	)

	// Encode one packet up front so every decoder has valid input to chew on.
	enc, err := NewEncoder(concurrentSampleRate, concurrentChannels, AppAudio)
	if err != nil {
		t.Fatalf("failed to create encoder: %v", err)
	}
	packet := make([]byte, 4000)
	n, err := enc.Encode(pcmTone(concurrentFrameSize, 0), packet)
	if err != nil {
		t.Fatalf("failed to encode seed packet: %v", err)
	}
	packet = packet[:n]

	var wg sync.WaitGroup
	errs := make(chan error, decoders)

	for d := 0; d < decoders; d++ {
		dec, err := NewDecoder(concurrentSampleRate, concurrentChannels)
		if err != nil {
			t.Fatalf("decoder %d: failed to create: %v", d, err)
		}

		wg.Add(1)
		go func(dec *Decoder) {
			defer wg.Done()
			pcm := make([]int16, concurrentFrameSize)
			for i := 0; i < framesPerDecoder; i++ {
				if _, err := dec.Decode(packet, pcm); err != nil {
					errs <- err
					return
				}
			}
		}(dec)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent decode failed: %v", err)
	}
}

// TestEncodeDecodeConcurrent mixes both directions, since encoders and decoders
// share the same malloc/free functions and the same linear memory.
func TestEncodeDecodeConcurrent(t *testing.T) {
	const rounds = 250

	enc0, err := NewEncoder(concurrentSampleRate, concurrentChannels, AppAudio)
	if err != nil {
		t.Fatalf("failed to create seed encoder: %v", err)
	}
	packet := make([]byte, 4000)
	n, err := enc0.Encode(pcmTone(concurrentFrameSize, 0), packet)
	if err != nil {
		t.Fatalf("failed to encode seed packet: %v", err)
	}
	packet = packet[:n]

	enc, err := NewEncoder(concurrentSampleRate, concurrentChannels, AppAudio)
	if err != nil {
		t.Fatalf("failed to create encoder: %v", err)
	}
	dec, err := NewDecoder(concurrentSampleRate, concurrentChannels)
	if err != nil {
		t.Fatalf("failed to create decoder: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 2)

	wg.Add(1)
	go func() {
		defer wg.Done()
		pcm := pcmTone(concurrentFrameSize, 11)
		out := make([]byte, 4000)
		for i := 0; i < rounds; i++ {
			if _, err := enc.Encode(pcm, out); err != nil {
				errs <- err
				return
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		pcm := make([]int16, concurrentFrameSize)
		for i := 0; i < rounds; i++ {
			if _, err := dec.Decode(packet, pcm); err != nil {
				errs <- err
				return
			}
		}
	}()

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent encode/decode failed: %v", err)
	}
}
