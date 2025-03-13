# Для зображення
boxes = detect_players_image('image.jpg', 'weights/best.pt')
print(boxes)

# Для відео
detect_players_video('test.mp4', 'weights/best.pt', 'output_tracked.mp4')
