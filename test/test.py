from pyvi import ViTokenizer

str = 'Phạm Lịch đưa ra tin nhắn        cũng là câu trả lời cuối cùng dành cho những người vào chửi bới xúc phạm cô khi lên tiếng tố ca sĩ Phạm Anh Khoa gạ tình. Phạm Lịch cho biết, đây cũng là lần cuối cùng cô nhắc về vấn đề này, cô nói: “Lịch muốn giải quyết vấn đề này từ sớm rồi nhưng có vài vấn đề đã xảy ra. Sẽ chẳng ai cảm nhận được rõ nhất nếu không đứng ở vị trí của Lịch và một lần nữa cám ơn mọi người đã bên cạnh Lịch trong thời gian vừa qua. Sau khi nói chuyện lần cuối và muốn có một lời xin lỗi từ phía anh Khoa nhưng ngược lại là sự coi thường và xúc phạm Lịch mới quyết định lên tiếng”. Toàn bộ nội dung cuộc trò chuyện của Phạm Anh Khoa và Phạm Lịch'
token = ViTokenizer.tokenize(str.strip())
print(token.strip().split())

