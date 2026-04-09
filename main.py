from PIL import Image, ImageDraw
import os

def generate_elite_card():
    # Lienzo vertical elegante (1000x1350)
    img = Image.new('RGB', (1000, 1350), color=(5, 5, 10))
    draw = ImageDraw.Draw(img)
    
    # ENCABEZADO DORADO (ESTILO BETFAIR)
    draw.rectangle([0, 0, 1000, 120], fill="#FFB80C")
    draw.text((500, 60), "CARTELERA DE ÉLITE - 09/04/2026", fill="black", anchor="mm")

    # SECCIÓN CHAMPIONS LEAGUE
    draw.rectangle([50, 150, 950, 200], fill="#000080")
    draw.text((500, 175), "⚽ UEFA CHAMPIONS LEAGUE (QUARTER-FINALS)", fill="white", anchor="mm")
    
    ucl_data = [
        ("Man City vs Real Madrid", "Gana Man City", "58.0%", "1.68"),
        ("Arsenal vs Bayern", "Gana Arsenal", "48.0%", "1.85"),
        ("City vs Madrid", "Ambos Anotan (GG)", "74.0%", "1.65")
    ]
    
    y = 230
    for match, pick, prob, odds in ucl_data:
        draw.text((100, y), match, fill="white")
        draw.text((100, y+35), f"Pick: {pick}", fill="#FFB80C")
        draw.text((700, y), f"Prob: {prob}", fill="#00FF00")
        draw.text((700, y+35), f"Odds BF: {odds}", fill="gray")
        draw.line([100, y+85, 900, y+85], fill=(40, 40, 40))
        y += 110

    # SECCIÓN NBA
    y += 40
    draw.rectangle([50, y, 950, y+50], fill="#CC5500")
    draw.text((500, y+25), "🏀 NBA ELITE MATCHUPS", fill="white", anchor="mm")
    
    nba_data = [
        ("Denver vs Memphis", "Gana Denver (ML)", "82.5%", "1.25"),
        ("NY Knicks vs ATL Hawks", "Gana Knicks (ML)", "71.0%", "1.68"),
        ("Denver Nuggets", "Hándicap -8.5", "62.0%", "1.91")
    ]
    
    y += 80
    for match, pick, prob, odds in nba_data:
        draw.text((100, y), match, fill="white")
        draw.text((100, y+35), f"Pick: {pick}", fill="#FFB80C")
        draw.text((700, y), f"Prob: {prob}", fill="#00FF00")
        draw.text((700, y+35), f"Odds BF: {odds}", fill="gray")
        draw.line([100, y+85, 900, y+85], fill=(40, 40, 40))
        y += 110

    # RESUMEN FINANCIERO MXN
    draw.rectangle([50, 1180, 950, 1300], outline="#FFB80C", width=3)
    draw.text((500, 1215), "INVERSIÓN TOTAL HOY: $4,700.00 MXN", fill="white", anchor="mm")
    draw.text((500, 1265), f"BANKROLL DISPONIBLE: $3,410.00 MXN", fill="#00FF00", anchor="mm")

    img.save("prediction_card.png")
    print("✅ Cartelera de Élite generada.")

if __name__ == "__main__":
    generate_elite_card()
